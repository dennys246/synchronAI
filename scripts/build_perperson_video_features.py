#!/usr/bin/env python3
"""Construct per-person DINOv2 features from patch features + pose bboxes.

For each 1-second segment, combine:
  - Patch features (12, 257, 768) from data/dinov2_features_patches/
  - Pose keypoints (12, 2, 17, 3) from data/pose_features/
to produce a per-person feature tensor of shape (12, 2, 768).

For each frame, per person:
  1. Derive a bounding box from the high-confidence keypoints (min/max of x, y
     across keypoints with confidence ≥ MIN_KPT_CONF).
  2. Identify the patches whose centers fall inside the bbox. DINOv2-base at
     224² uses a 16×16 patch grid plus 1 CLS; we ignore CLS (index 0) and
     work with patches 1..256.
  3. Mean-pool the matching patch features → (768,) per person per frame.
  4. Fallback when no keypoint passes the threshold (person missing this
     frame): use the CLS token feature for that frame. This keeps the slot
     informative without zeroing it.

This is the cheaper alternative to re-extracting DINOv2 from per-person
crops. The approximation is "patches inside bbox" vs. "DINOv2 run on the
crop image" — coarser, but uses only data already on disk. If
cross-person attention helps with these features, a follow-up extraction
with per-crop DINOv2 is justified.

Output:
    data/perperson_video_features/
        feature_index.csv               # Same schema as DINOv2/WavLM/pose
        features/{stem}_{second}_{hash}.pt   # (12, 2, 768) float32
        bboxes/{stem}_{second}_{hash}.pt     # (12, 2, 4) float32, normalized
                                             #   coords or FALLBACK_BBOX = -1
        fallback_per_segment.csv        # per-(segment, slot) diagnostic rows
        fallback_per_subject.csv        # per-(subject, session) aggregate

The diagnostic CSVs make it trivial to find which subjects/sessions are
driving the global fallback rate, so we know where the pose-detection
pipeline is failing (e.g., specific recording angles or activities).

Usage:
    python scripts/build_perperson_video_features.py \\
        --patch-feature-dir data/dinov2_features_patches \\
        --pose-feature-dir  data/pose_features \\
        --output-dir        data/perperson_video_features_conf005 \\
        --min-kpt-conf      0.05
"""

import argparse
import csv
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


SCRIPT_VERSION = "build_perperson_video_features-v1"

# DINOv2-base at 224x224: 16x16 patches plus 1 CLS token at index 0.
N_GRID = 16  # 16x16 = 256 patches
N_TOKENS = N_GRID * N_GRID + 1  # 257


def feature_filename(video_path: str, second: int) -> str:
    """Match the schema used by the other feature extractors."""
    key = f"{video_path}:{second}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    stem = Path(video_path).stem[:20]
    return f"{stem}_{second:05d}_{h}.pt"


def _patch_centers():
    """Return (256, 2) array of patch (x, y) centers in normalized [0, 1] coords.

    Patch (i, j) for i in [0..15] (column = x-direction), j in [0..15]
    (row = y-direction) covers the image region with center at
    ((i + 0.5) / 16, (j + 0.5) / 16).
    """
    grid = np.linspace(0.5 / N_GRID, 1 - 0.5 / N_GRID, N_GRID, dtype=np.float32)
    xs, ys = np.meshgrid(grid, grid, indexing="xy")
    return np.stack([xs.flatten(), ys.flatten()], axis=-1)  # (256, 2)


PATCH_CENTERS = _patch_centers()  # cached at import


def derive_bbox(kpts: np.ndarray, min_conf: float) -> tuple[float, float, float, float] | None:
    """Per-person bbox from high-confidence keypoints.

    Args:
        kpts: (17, 3) array of (x_norm, y_norm, confidence) for one person.
        min_conf: keypoint confidence threshold.

    Returns:
        (x1, y1, x2, y2) in normalized [0, 1] coords, or None if no
        keypoint passes the threshold.
    """
    mask = kpts[:, 2] >= min_conf
    if not mask.any():
        return None
    xs = kpts[mask, 0]
    ys = kpts[mask, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    # Pad the bbox slightly so small / cramped detections still capture context.
    pad = 0.03
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(1.0, x2 + pad)
    y2 = min(1.0, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def pool_patches_in_bbox(patches: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Mean-pool patches whose centers fall inside bbox.

    Args:
        patches: (256, 768) non-CLS patch tokens for one frame.
        bbox: (x1, y1, x2, y2) normalized coords.

    Returns:
        (768,) pooled vector. If no patches qualify, returns the all-patches mean
        (full-frame fallback for sanity).
    """
    x1, y1, x2, y2 = bbox
    cx = PATCH_CENTERS[:, 0]
    cy = PATCH_CENTERS[:, 1]
    mask = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    if mask.any():
        return patches[mask].mean(axis=0)
    # Fallback: use the global patch mean (i.e. v2-style meanpatch) for this
    # frame. Rare — only triggers when bbox is smaller than a single patch.
    return patches.mean(axis=0)


FALLBACK_BBOX = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def build_one(
    patch_feature: torch.Tensor,
    pose_feature: torch.Tensor,
    min_kpt_conf: float,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """Construct (T, 2, D) per-person features + (T, 2, 4) bbox record.

    Args:
        patch_feature: (T, 257, D) DINOv2 patch tokens (1 CLS + 256 patches).
        pose_feature: (T, 2, 17, 3) YOLO26 keypoints, (x_norm, y_norm, conf).
        min_kpt_conf: keypoint confidence threshold for bbox derivation.

    Returns:
        Tuple of:
          - features: (T, 2, D) tensor of per-person pooled features.
          - bboxes: (T, 2, 4) np.ndarray. Each row is (x1, y1, x2, y2) for the
            derived bbox, or FALLBACK_BBOX = (-1, -1, -1, -1) when the slot
            fell back to the CLS token. Persisted alongside features to
            support per-frame fallback diagnostics + visual debugging.
          - diag: dict of per-slot counts:
              - fallback_to_cls_slot{0,1}: frames in this segment that used CLS
              - n_kpts_above_thresh_slot{0,1}: list of len T with per-frame keypoint counts
    """
    T = patch_feature.shape[0]
    D = patch_feature.shape[-1]
    patches_np = patch_feature[:, 1:, :].numpy()  # (T, 256, D)
    cls_np = patch_feature[:, 0, :].numpy()       # (T, D)
    pose_np = pose_feature.numpy()                # (T, 2, 17, 3)

    out = np.zeros((T, 2, D), dtype=np.float32)
    bboxes = np.tile(FALLBACK_BBOX, (T, 2, 1)).astype(np.float32)
    fallback_to_cls = [0, 0]
    n_kpts_above = [[0] * T, [0] * T]

    for t in range(T):
        for slot in range(2):
            kpts = pose_np[t, slot]
            n_kpts_above[slot][t] = int((kpts[:, 2] >= min_kpt_conf).sum())
            bbox = derive_bbox(kpts, min_kpt_conf)
            if bbox is None:
                # Person missing in this frame — use CLS token as the
                # per-person feature. CLS encodes scene-level semantics and
                # is the least-bad single-vector fallback for "no body
                # detected this frame." bbox stays FALLBACK_BBOX.
                out[t, slot] = cls_np[t]
                fallback_to_cls[slot] += 1
            else:
                out[t, slot] = pool_patches_in_bbox(patches_np[t], bbox)
                bboxes[t, slot] = np.array(bbox, dtype=np.float32)

    diag = {
        "fallback_to_cls_slot0": fallback_to_cls[0],
        "fallback_to_cls_slot1": fallback_to_cls[1],
        "n_kpts_above_slot0": n_kpts_above[0],  # list of len T
        "n_kpts_above_slot1": n_kpts_above[1],
    }
    return torch.from_numpy(out), bboxes, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--patch-feature-dir", type=Path, required=True,
        help="data/dinov2_features_patches",
    )
    parser.add_argument(
        "--pose-feature-dir", type=Path, required=True,
        help="data/pose_features",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="data/perperson_video_features",
    )
    parser.add_argument(
        "--min-kpt-conf", type=float, default=0.25,
        help="Minimum keypoint confidence to include in bbox derivation.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N samples (smoke test).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples already in the output index.",
    )
    args = parser.parse_args()

    logger.info(f"=== [{SCRIPT_VERSION}] ===")
    logger.info(f"Patch features: {args.patch_feature_dir}")
    logger.info(f"Pose features:  {args.pose_feature_dir}")
    logger.info(f"Output:         {args.output_dir}")
    logger.info(f"Min kpt conf:   {args.min_kpt_conf}")

    output_dir = args.output_dir
    features_dir = output_dir / "features"
    bboxes_dir = output_dir / "bboxes"
    features_dir.mkdir(parents=True, exist_ok=True)
    bboxes_dir.mkdir(parents=True, exist_ok=True)
    fallback_csv_path = output_dir / "fallback_per_segment.csv"

    # Load + inner-join the two indices on (video_path, second).
    patch_idx_path = args.patch_feature_dir / "feature_index.csv"
    pose_idx_path = args.pose_feature_dir / "feature_index.csv"
    if not patch_idx_path.exists():
        logger.error(f"Missing patch index: {patch_idx_path}")
        sys.exit(1)
    if not pose_idx_path.exists():
        logger.error(f"Missing pose index: {pose_idx_path}")
        sys.exit(1)

    with open(patch_idx_path) as f:
        patch_rows = {(r["video_path"], int(float(r["second"]))): r for r in csv.DictReader(f)}
    with open(pose_idx_path) as f:
        pose_rows = {(r["video_path"], int(float(r["second"]))): r for r in csv.DictReader(f)}

    common = sorted(set(patch_rows.keys()) & set(pose_rows.keys()))
    logger.info(
        f"Patch: {len(patch_rows)}, Pose: {len(pose_rows)}, Joined: {len(common)} samples"
    )
    if args.limit is not None:
        common = common[: args.limit]
        logger.info(f"Limit: processing only first {len(common)} samples")

    # Resume support.
    existing = set()
    index_file = output_dir / "feature_index.csv"
    if args.skip_existing and index_file.exists():
        with open(index_file) as f:
            for r in csv.DictReader(f):
                if (features_dir / r["feature_file"]).exists():
                    existing.add((r["video_path"], int(float(r["second"]))))
        logger.info(f"Resume: {len(existing)} samples already done, will skip")

    results = []
    fallback_rows = []  # one row per (segment, slot) — drives the deep-dive analysis
    extracted = 0
    skipped = 0
    errors = 0
    total_fallbacks = [0, 0]
    start = time.time()
    patch_dir = args.patch_feature_dir / "features"
    pose_dir = args.pose_feature_dir / "features"

    for idx, key in enumerate(common):
        video_path, second = key
        if key in existing:
            skipped += 1
            patch_row = patch_rows[key]
            fname = feature_filename(video_path, second)
            results.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": int(float(patch_row["label"])),
                "subject_id": patch_row.get("subject_id", ""),
                "session": patch_row.get("session", ""),
                "feature_dim": 768,
                "n_frames": 12,
                "n_persons": 2,
            })
            continue
        try:
            patch_row = patch_rows[key]
            pose_row = pose_rows[key]
            patch_t = torch.load(
                patch_dir / patch_row["feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            pose_t = torch.load(
                pose_dir / pose_row["feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            features, bboxes, diag = build_one(patch_t, pose_t, args.min_kpt_conf)
            fname = feature_filename(video_path, second)
            torch.save(features, features_dir / fname)
            torch.save(torch.from_numpy(bboxes), bboxes_dir / fname)
            extracted += 1
            total_fallbacks[0] += diag["fallback_to_cls_slot0"]
            total_fallbacks[1] += diag["fallback_to_cls_slot1"]
            results.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": int(float(patch_row["label"])),
                "subject_id": patch_row.get("subject_id", ""),
                "session": patch_row.get("session", ""),
                "feature_dim": 768,
                "n_frames": 12,
                "n_persons": 2,
            })
            # Per-segment, per-slot fallback diagnostics. With 12 frames per
            # segment, two rows captures everything we need for the deep dive
            # (n frames that fell back; min/max/mean keypoint counts).
            subj = patch_row.get("subject_id", "")
            sess = patch_row.get("session", "")
            for slot in range(2):
                kpts_seq = diag[f"n_kpts_above_slot{slot}"]
                fallback_rows.append({
                    "feature_file": fname,
                    "subject_id": subj,
                    "session": sess,
                    "video_path": video_path,
                    "second": second,
                    "slot": slot,
                    "n_fallback_frames": diag[f"fallback_to_cls_slot{slot}"],
                    "n_total_frames": 12,
                    "min_kpts_above_thresh": int(min(kpts_seq)),
                    "mean_kpts_above_thresh": float(sum(kpts_seq) / len(kpts_seq)),
                    "max_kpts_above_thresh": int(max(kpts_seq)),
                })
        except Exception as e:
            errors += 1
            logger.warning(f"Error on {video_path} @ {second}: {e}")
            continue

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (extracted + skipped) / elapsed if elapsed > 0 else 0
            eta = (len(common) - idx - 1) / rate / 60 if rate > 0 else 0
            logger.info(
                f"  {idx+1}/{len(common)} ({rate:.1f}/s, ETA {eta:.0f} min). "
                f"Extracted {extracted}, skipped {skipped}, errors {errors}. "
                f"Fallback-to-CLS: slot0={total_fallbacks[0]} slot1={total_fallbacks[1]}"
            )
            _write_index(index_file, results)
            _write_fallback_csv(fallback_csv_path, fallback_rows)

    _write_index(index_file, results)
    _write_fallback_csv(fallback_csv_path, fallback_rows)
    elapsed = time.time() - start
    logger.info(
        f"Done in {elapsed/60:.1f} min. Extracted {extracted}, "
        f"skipped {skipped}, errors {errors}."
    )
    total_frames = 12 * extracted
    if total_frames > 0:
        logger.info(
            f"Fallback-to-CLS rate: "
            f"slot0={total_fallbacks[0]/total_frames*100:.1f}%, "
            f"slot1={total_fallbacks[1]/total_frames*100:.1f}% "
            f"(high % means many frames had no detected person in that slot)"
        )
        _write_subject_summary(output_dir / "fallback_per_subject.csv", fallback_rows)
        logger.info(f"Per-segment diagnostics:  {fallback_csv_path}")
        logger.info(f"Per-subject diagnostics:  {output_dir}/fallback_per_subject.csv")
        logger.info(f"Bboxes:                   {bboxes_dir}/")


def _write_index(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_fallback_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_subject_summary(path: Path, fallback_rows: list[dict]) -> None:
    """Aggregate per-(subject, session) fallback rates from the per-segment rows."""
    if not fallback_rows:
        return
    from collections import defaultdict
    agg = defaultdict(lambda: {
        "n_segments": 0,
        "n_fallback_frames_slot0": 0,
        "n_fallback_frames_slot1": 0,
        "n_total_frames": 0,
    })
    for r in fallback_rows:
        key = (r["subject_id"], r["session"])
        a = agg[key]
        if r["slot"] == 0:
            # Each segment yields two fallback_rows (slot 0, slot 1). Count
            # segments + total frames once per segment, on slot 0.
            a["n_segments"] += 1
            a["n_total_frames"] += r["n_total_frames"]
            a["n_fallback_frames_slot0"] += r["n_fallback_frames"]
        else:
            a["n_fallback_frames_slot1"] += r["n_fallback_frames"]

    rows = []
    for (subj, sess), a in sorted(agg.items()):
        ntotal = a["n_total_frames"] or 1
        rows.append({
            "subject_id": subj,
            "session": sess,
            "n_segments": a["n_segments"],
            "n_total_frames": a["n_total_frames"],
            "n_fallback_slot0": a["n_fallback_frames_slot0"],
            "n_fallback_slot1": a["n_fallback_frames_slot1"],
            "fallback_rate_slot0": round(a["n_fallback_frames_slot0"] / ntotal, 4),
            "fallback_rate_slot1": round(a["n_fallback_frames_slot1"] / ntotal, 4),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

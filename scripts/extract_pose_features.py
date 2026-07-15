#!/usr/bin/env python3
"""Extract per-frame 2-person pose keypoints for dyadic videos at 12fps.

Probe 3 in the multimodal-ceiling investigation: DINOv2 and WavLM are
scene/speech-encoded; neither captures inter-person geometry directly.
Per-person pose keypoints expose relational signal (body orientation,
gesture coupling, gaze proxies) that scene-pretrained backbones don't.

Mirrors the DINOv2/WavLM feature pipeline:
  - Samples 12 frames per 1-second window
  - Writes per-(video, second) .pt files + feature_index.csv
  - Same join keys (video_path, second) so the trainer can fuse with
    DINOv2 / WavLM via the existing merge_feature_indices logic.

Output per file: torch tensor of shape (12, 2, 17, 3) — float32.
  Axes: (frame, person_slot, keypoint, channel)
  Channels: (x_normalized, y_normalized, confidence)
  Coordinates normalized to [0, 1] by frame width/height.
  Missing person/keypoint slots are zero-filled (confidence=0).

Person identity (slot 0 vs slot 1) is tracked across the 12 frames within
each 1-second window via greedy IoU matching. Identity is NOT preserved
across seconds — each segment's slot assignment is independent, so
downstream models should not rely on consistent person-A vs person-B
identity across consecutive seconds.

Uses YOLO26n-pose (Ultralytics, released Jan 2026): single forward pass
gives bbox + 17 COCO keypoints per detected person. Top-2 detections per
frame by confidence.

Usage:
    python scripts/extract_pose_features.py \\
        --labels-file data/labels.csv \\
        --output-dir data/pose_features/ \\
        --model-weights scripts/yolo26n-pose.pt \\
        --sample-fps 12 \\
        --device cpu
"""

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


SCRIPT_VERSION = "extract_pose_features-v1"


def feature_filename(video_path: str, second: int) -> str:
    """Stable per-(video, second) filename matching DINOv2/WavLM scheme."""
    base = Path(video_path).stem
    h = hashlib.md5(f"{video_path}_{second}".encode()).hexdigest()[:12]
    return f"{base}_{second:05d}_{h}.pt"


def iou(b1, b2) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter + 1e-8)


def read_window_frames(
    video_path: str,
    second: int,
    sample_fps: float = 12.0,
    window_seconds: float = 1.0,
):
    """Read `sample_fps * window_seconds` frames from the given second.

    Returns: (frames_list, frame_height, frame_width). frames_list is a
    list of BGR uint8 arrays (one per sampled frame), zero-padded if the
    video ends mid-window.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return None, None, None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    n_frames = int(sample_fps * window_seconds)
    target_frame_indices = [
        int((second + i / sample_fps) * fps)
        for i in range(n_frames)
    ]
    frames = []
    for fidx in target_frame_indices:
        if fidx >= frame_count:
            frames.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
        else:
            frames.append(frame)
    cap.release()
    return frames, h, w


def run_pose_on_window(model, frames, frame_h, frame_w, conf_threshold=0.25):
    """Run YOLO26-pose on each frame in the window.

    Returns: list of length n_frames, each element a list of detections
    [{"bbox": [x1,y1,x2,y2], "conf": float, "kpts": (17, 3)}, ...].
    Frames with missing video data → empty list.
    """
    per_frame = []
    for frame in frames:
        if frame is None:
            per_frame.append([])
            continue
        # YOLO26 accepts BGR ndarray. verbose=False to silence per-frame logs.
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        dets = []
        if not results:
            per_frame.append(dets)
            continue
        res = results[0]
        if res.boxes is None or res.keypoints is None:
            per_frame.append(dets)
            continue
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes.xyxy is not None else None
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
        # Ultralytics keypoints: .data shape (n_persons, 17, 3) where channels
        # are (x_pixel, y_pixel, confidence).
        kpts_all = res.keypoints.data.cpu().numpy() if res.keypoints.data is not None else None
        if boxes is None or kpts_all is None:
            per_frame.append(dets)
            continue
        n = len(boxes)
        for i in range(n):
            dets.append({
                "bbox": boxes[i].tolist(),
                "conf": float(confs[i]) if confs is not None else 0.0,
                "kpts": kpts_all[i],  # (17, 3) pixel coords + conf
            })
        per_frame.append(dets)
    return per_frame


def assign_two_person_slots(per_frame_dets, frame_h, frame_w):
    """Greedy IoU-based 2-person identity tracking across the window.

    Returns: numpy float32 array of shape (n_frames, 2, 17, 3).
    Channels: (x_normalized, y_normalized, confidence).
    """
    n_frames = len(per_frame_dets)
    out = np.zeros((n_frames, 2, 17, 3), dtype=np.float32)
    if frame_h <= 0 or frame_w <= 0:
        return out

    # Per-slot most recent bbox (for IoU matching). None until a person occupies the slot.
    prev_boxes = [None, None]

    for f_idx, frame_dets in enumerate(per_frame_dets):
        if not frame_dets:
            continue
        # Top 2 detections by confidence (we only ever care about a dyad).
        sorted_dets = sorted(frame_dets, key=lambda d: -d["conf"])[:2]

        if prev_boxes[0] is None and prev_boxes[1] is None:
            # First frame with detections: assign in confidence order.
            for slot, det in enumerate(sorted_dets):
                _write_slot(out, f_idx, slot, det, frame_h, frame_w)
                prev_boxes[slot] = det["bbox"]
            continue

        # Greedy IoU match: for each detection, assign to the most-IoU prev slot.
        assigned_slots = set()
        unassigned_dets = []
        for det in sorted_dets:
            best_slot = -1
            best_iou = 0.0
            for slot in range(2):
                if slot in assigned_slots or prev_boxes[slot] is None:
                    continue
                cur = iou(prev_boxes[slot], det["bbox"])
                if cur > best_iou:
                    best_iou = cur
                    best_slot = slot
            if best_slot >= 0 and best_iou > 0.2:
                _write_slot(out, f_idx, best_slot, det, frame_h, frame_w)
                prev_boxes[best_slot] = det["bbox"]
                assigned_slots.add(best_slot)
            else:
                unassigned_dets.append(det)

        # Unmatched detections fill any empty slot.
        for det in unassigned_dets:
            for slot in range(2):
                if slot not in assigned_slots:
                    _write_slot(out, f_idx, slot, det, frame_h, frame_w)
                    prev_boxes[slot] = det["bbox"]
                    assigned_slots.add(slot)
                    break

    return out


def _write_slot(out, f_idx, slot, det, frame_h, frame_w):
    kpts = det["kpts"]  # (17, 3)
    out[f_idx, slot, :, 0] = kpts[:, 0] / float(frame_w)  # x normalized
    out[f_idx, slot, :, 1] = kpts[:, 1] / float(frame_h)  # y normalized
    out[f_idx, slot, :, 2] = kpts[:, 2]  # confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels-file", required=True, help="Path to labels.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--model-weights",
        default="scripts/yolo26n-pose.pt",
        help="Path to YOLO26-pose weights (.pt). Defaults to scripts/yolo26n-pose.pt.",
    )
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N labels (for smoke-test runs).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip windows whose .pt is already on disk.",
    )
    args = parser.parse_args()

    logger.info(f"=== [{SCRIPT_VERSION}] ===")
    logger.info(f"Labels:  {args.labels_file}")
    logger.info(f"Output:  {args.output_dir}")
    logger.info(f"Model:   {args.model_weights}")
    logger.info(f"Sample:  {args.sample_fps} fps × {args.window_seconds}s = "
                f"{int(args.sample_fps * args.window_seconds)} frames/segment")
    logger.info(f"Device:  {args.device}")

    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports so the script's --help works without ultralytics installed.
    import cv2  # noqa: F401  (used by read_window_frames; fail early if missing)
    from ultralytics import YOLO
    import torch

    logger.info(f"Loading YOLO26-pose from {args.model_weights}")
    model = YOLO(args.model_weights)
    if args.device != "cpu":
        model.to(args.device)

    # Read labels.csv.
    with open(args.labels_file) as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]
    logger.info(f"Loaded {len(rows)} labels.")

    # Resume support.
    existing = set()
    index_file = output_dir / "feature_index.csv"
    if index_file.exists() and args.skip_existing:
        with open(index_file) as f:
            for r in csv.DictReader(f):
                fp = features_dir / r["feature_file"]
                if fp.exists():
                    existing.add((r["video_path"], int(float(r["second"]))))
        logger.info(f"Resume: {len(existing)} segments already on disk, will skip.")

    n_frames = int(args.sample_fps * args.window_seconds)
    results_for_index = []
    errors = 0
    extracted = 0
    skipped = 0
    start = time.time()

    for idx, row in enumerate(rows):
        video_path = row["video_path"]
        second = int(float(row["second"]))
        fname = feature_filename(video_path, second)

        if (video_path, second) in existing:
            skipped += 1
            results_for_index.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": int(float(row["label"])),
                "subject_id": row.get("subject_id", ""),
                "session": row.get("session", ""),
                "feature_dim": 2 * 17 * 3,  # nominal per-frame dim
                "n_frames": n_frames,
            })
            continue

        try:
            frames, frame_h, frame_w = read_window_frames(
                video_path, second,
                sample_fps=args.sample_fps,
                window_seconds=args.window_seconds,
            )
            if frames is None:
                logger.warning(f"Failed to read {video_path} @ second {second}")
                errors += 1
                continue

            per_frame_dets = run_pose_on_window(
                model, frames, frame_h, frame_w,
                conf_threshold=args.conf_threshold,
            )
            tensor = assign_two_person_slots(per_frame_dets, frame_h, frame_w)

            # Ensure exact n_frames by zero-padding or truncating.
            if tensor.shape[0] < n_frames:
                pad = np.zeros((n_frames - tensor.shape[0], 2, 17, 3), dtype=np.float32)
                tensor = np.concatenate([tensor, pad], axis=0)
            elif tensor.shape[0] > n_frames:
                tensor = tensor[:n_frames]

            torch.save(torch.from_numpy(tensor), features_dir / fname)
            extracted += 1
            results_for_index.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": int(float(row["label"])),
                "subject_id": row.get("subject_id", ""),
                "session": row.get("session", ""),
                "feature_dim": 2 * 17 * 3,
                "n_frames": n_frames,
            })
        except Exception as e:
            logger.error(f"Error on {video_path} @ {second}: {e}")
            errors += 1
            continue

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (extracted + skipped) / elapsed if elapsed > 0 else 0
            eta = (len(rows) - idx - 1) / rate / 60 if rate > 0 else 0
            logger.info(
                f"  Processed {idx+1}/{len(rows)} "
                f"(extracted {extracted}, skipped {skipped}, errors {errors}). "
                f"Rate: {rate:.1f}/s, ETA: {eta:.1f} min"
            )
            # Periodic index flush so we don't lose progress on a crash.
            _write_index(index_file, results_for_index)

    _write_index(index_file, results_for_index)
    elapsed = time.time() - start
    logger.info(
        f"Done in {elapsed/60:.1f} min. "
        f"Extracted: {extracted}, skipped: {skipped}, errors: {errors}."
    )
    logger.info(f"Feature index: {index_file}")


def _write_index(index_file: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(index_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

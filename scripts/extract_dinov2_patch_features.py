#!/usr/bin/env python3
"""Extract DINOv2 patch-grid features (full token sequence) per video window.

Companion to extract_dinov2_features.py — that script saves pooled features
(CLS or mean-of-patches) of shape (n_frames, D). This one saves the full
per-token sequence of shape (n_frames, 1 + n_patches, D), which preserves
spatial information for downstream probes that need it (probe 1: spatial
attention over patches).

Why this exists as a separate script: data/dinov2_features/ on disk holds
2D pooled features despite the dir name — the prior extraction sliced
the CLS token without .clone(), so torch.save serialized the 2D view but
inherited the 3D storage. Resulting files are 9.5MB each but load as
shape (12, 768). Trying to retrofit the existing script invites the same
bug to come back; cleaner to keep the patch-grid path isolated and use
encoder.forward_patches() (which clones).

Usage:
    python scripts/extract_dinov2_patch_features.py \\
        --labels-file data/labels.csv \\
        --output-dir data/dinov2_features_patches \\
        --backbone dinov2-base \\
        --sample-fps 12 \\
        --window-seconds 1.0 \\
        --frame-size 224

Output:
    data/dinov2_features_patches/
        feature_index.csv          # Maps feature files to metadata
        features/
            {stem}_{second}_{hash}.pt  # Each: (n_frames, 1+n_patches, D)
                                       # For dinov2-base @ 224: (12, 257, 768)
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synchronai.data.video.processing import (
    VideoReaderPool,
    read_window_frames_dinov2,
)
from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def feature_filename(video_path: str, second: int) -> str:
    """Same scheme as extract_dinov2_features.py — shared join key for the trainer."""
    key = f"{video_path}:{second}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    video_stem = Path(video_path).stem[:20]
    return f"{video_stem}_{second:05d}_{h}.pt"


def extract(
    labels_file: str,
    output_dir: str,
    backbone: str = "dinov2-base",
    sample_fps: float = 12.0,
    window_seconds: float = 1.0,
    frame_size: int = 224,
    device: str = "cpu",
) -> None:
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading labels from {labels_file}")
    df = pd.read_csv(labels_file)
    logger.info(f"Found {len(df)} windows across {df['video_path'].nunique()} videos")

    # Resume support: reuse the index of any previously-extracted patch files.
    index_file = output_dir / "feature_index.csv"
    existing = set()
    if index_file.exists():
        existing_df = pd.read_csv(index_file)
        for _, row in existing_df.iterrows():
            if (features_dir / row["feature_file"]).exists():
                existing.add((row["video_path"], int(row["second"])))
        logger.info(f"Resume: {len(existing)} segments already on disk, will skip.")

    logger.info(f"Loading DINOv2 backbone: {backbone}")
    # pool_mode is irrelevant here — we call forward_patches() directly,
    # which bypasses the pooling branch in forward(). Use 'cls' to keep
    # the cached-encoder key consistent with prior extractions.
    encoder = DINOv2FeatureExtractor(
        model_name=backbone,
        device=device,
        freeze=True,
        pool_mode="cls",
    )
    encoder._load_model()
    encoder.eval()
    feature_dim = encoder.feature_dim
    n_frames = int(sample_fps * window_seconds)

    # Discover patch count via a single throwaway forward pass.
    probe = torch.zeros((1, 3, frame_size, frame_size), dtype=torch.float32)
    with torch.no_grad():
        probe_out = encoder.forward_patches(probe)
    n_tokens = probe_out.shape[1]
    logger.info(
        f"Token count per frame: {n_tokens} (1 CLS + {n_tokens - 1} patches). "
        f"Feature dim per token: {feature_dim}."
    )
    logger.info(f"Frames per window: {n_frames} ({sample_fps} fps × {window_seconds}s)")

    reader_pool = VideoReaderPool(max_readers=4)

    results = []
    skipped = 0
    errors = 0
    extracted = 0
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting patch features"):
        video_path = row["video_path"]
        second = int(row["second"])
        fname = feature_filename(video_path, second)

        if (video_path, second) in existing:
            skipped += 1
            results.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": int(row["label"]),
                "subject_id": row.get("subject_id", ""),
                "session": row.get("session", ""),
                "feature_dim": feature_dim,
                "n_frames": n_frames,
                "n_tokens": n_tokens,
            })
            continue

        try:
            reader = reader_pool.get_reader(video_path)
            frames = read_window_frames_dinov2(
                video_path=video_path,
                second=second,
                sample_fps=sample_fps,
                window_seconds=window_seconds,
                target_size=frame_size,
                reader=reader,
            )

            frames_tensor = torch.from_numpy(frames).to(device)
            with torch.no_grad():
                features = encoder.forward_patches(frames_tensor)
            # features: (n_frames, n_tokens, feature_dim) — already .clone()'d
            # by forward_patches, so torch.save won't drag along extra storage.
            torch.save(features.cpu(), features_dir / fname)
            extracted += 1

            results.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": int(row["label"]),
                "subject_id": row.get("subject_id", ""),
                "session": row.get("session", ""),
                "feature_dim": feature_dim,
                "n_frames": n_frames,
                "n_tokens": n_tokens,
            })

        except Exception as e:
            errors += 1
            logger.warning(f"Error at {video_path} second {second}: {e}")
            continue

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = extracted / elapsed if elapsed > 0 else 0
            remaining = len(df) - idx - 1
            eta = remaining / rate / 60 if rate > 0 else 0
            logger.info(
                f"Progress: {idx + 1}/{len(df)} "
                f"({rate:.1f} windows/s, ~{eta:.0f} min remaining)"
            )
            # Flush index periodically so progress survives a crash.
            pd.DataFrame(results).to_csv(index_file, index=False)

    reader_pool.close_all()

    if results:
        pd.DataFrame(results).to_csv(index_file, index=False)
        logger.info(f"Saved feature index: {index_file} ({len(results)} entries)")

    elapsed = time.time() - start_time
    logger.info(
        f"Patch extraction complete in {elapsed / 60:.1f} min. "
        f"Extracted: {extracted}, Skipped: {skipped}, Errors: {errors}"
    )
    logger.info(f"Features saved to: {features_dir}")
    logger.info(f"Per-window shape: ({n_frames}, {n_tokens}, {feature_dim})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 patch-grid features (full token sequence)."
    )
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="dinov2-base")
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument(
        "--device", default="cpu",
        help="Device for DINOv2 inference (cpu, cuda).",
    )
    args = parser.parse_args()

    extract(
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        backbone=args.backbone,
        sample_fps=args.sample_fps,
        window_seconds=args.window_seconds,
        frame_size=args.frame_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

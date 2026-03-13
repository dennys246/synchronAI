#!/usr/bin/env python3
"""Extract DINOv2 features from video windows and save to disk.

This script pre-extracts frozen DINOv2 backbone features for all video windows
in a labels.csv file. Pre-extracted features enable fast CPU-based training of
the temporal aggregation (LSTM) and classification head without needing the
DINOv2 backbone loaded during training.

Usage:
    python scripts/extract_dinov2_features.py \
        --labels-file data/labels.csv \
        --output-dir data/dinov2_features \
        --backbone dinov2-base \
        --sample-fps 12 \
        --window-seconds 1.0 \
        --frame-size 224

Output:
    data/dinov2_features/
        feature_index.csv      # Maps feature files to metadata
        features/              # Directory of .pt feature tensors
            {stem}_{second}_{hash}.pt  # Each: (n_frames, feature_dim)
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
    load_video_info,
    read_window_frames_dinov2,
)
from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def feature_filename(video_path: str, second: int) -> str:
    """Generate a unique filename for a feature file."""
    key = f"{video_path}:{second}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    video_stem = Path(video_path).stem[:20]
    return f"{video_stem}_{second:05d}_{h}.pt"


def extract_features(
    labels_file: str,
    output_dir: str,
    backbone: str = "dinov2-base",
    sample_fps: float = 12.0,
    window_seconds: float = 1.0,
    frame_size: int = 224,
    device: str = "cpu",
    pool_mode: str = "mean_patch",
) -> None:
    """Extract DINOv2 features for all windows in a labels file.

    Args:
        labels_file: Path to labels.csv with video_path, second, label columns
        output_dir: Directory to save extracted features
        backbone: DINOv2 model variant (dinov2-small, dinov2-base, dinov2-large)
        sample_fps: Target frames per second for sampling
        window_seconds: Duration of each window in seconds
        frame_size: Frame size for DINOv2 preprocessing (224)
        device: Device for DINOv2 inference (cpu or cuda)
        pool_mode: Feature pooling mode ("cls" or "mean_patch")
    """
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    logger.info(f"Loading labels from {labels_file}")
    df = pd.read_csv(labels_file)
    logger.info(f"Found {len(df)} windows across {df['video_path'].nunique()} videos")

    # Check for existing features (resume support)
    index_file = output_dir / "feature_index.csv"
    existing_features = set()
    if index_file.exists():
        existing_df = pd.read_csv(index_file)
        for _, row in existing_df.iterrows():
            if (features_dir / row["feature_file"]).exists():
                existing_features.add((row["video_path"], int(row["second"])))
        logger.info(f"Found {len(existing_features)} existing features, will skip those")

    # Load DINOv2 model
    logger.info(f"Loading DINOv2 model: {backbone}")
    encoder = DINOv2FeatureExtractor(
        model_name=backbone,
        device=device,
        freeze=True,
        pool_mode=pool_mode,
    )
    encoder._load_model()
    encoder.eval()
    feature_dim = encoder.feature_dim
    n_frames = int(sample_fps * window_seconds)
    logger.info(f"DINOv2 loaded. Feature dim: {feature_dim}, device: {device}")
    logger.info(f"Frames per window: {n_frames} ({sample_fps} fps x {window_seconds}s)")

    # Create video reader pool
    reader_pool = VideoReaderPool(max_readers=4)

    # Process windows
    results = []
    skipped = 0
    errors = 0
    extracted = 0
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        video_path = row["video_path"]
        second = int(row["second"])
        fname = feature_filename(video_path, second)

        # Skip if already extracted
        if (video_path, second) in existing_features:
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
            })
            continue

        try:
            # Read and preprocess frames with DINOv2 normalization
            reader = reader_pool.get_reader(video_path)
            frames = read_window_frames_dinov2(
                video_path=video_path,
                second=second,
                sample_fps=sample_fps,
                window_seconds=window_seconds,
                target_size=frame_size,
                reader=reader,
            )

            # Extract features: (n_frames, C, H, W) → (n_frames, feature_dim)
            frames_tensor = torch.from_numpy(frames).to(device)
            with torch.no_grad():
                features = encoder(frames_tensor)

            # Save features to disk
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
            })

        except Exception as e:
            errors += 1
            logger.warning(f"Error at {video_path} second {second}: {e}")
            continue

        # Progress logging every 500 windows
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = extracted / elapsed if elapsed > 0 else 0
            remaining_windows = len(df) - idx - 1
            eta = remaining_windows / rate / 60 if rate > 0 else 0
            logger.info(
                f"Progress: {idx + 1}/{len(df)} "
                f"({rate:.1f} windows/s, ~{eta:.0f} min remaining)"
            )

    # Clean up
    reader_pool.close_all()

    # Save feature index
    if results:
        index_df = pd.DataFrame(results)
        index_df.to_csv(index_file, index=False)
        logger.info(f"Saved feature index: {index_file} ({len(index_df)} entries)")

    elapsed = time.time() - start_time
    logger.info(
        f"Feature extraction complete in {elapsed / 60:.1f} minutes. "
        f"Extracted: {extracted}, Skipped (existing): {skipped}, Errors: {errors}"
    )
    logger.info(f"Features saved to: {features_dir}")
    logger.info(f"Feature shape per window: ({n_frames}, {feature_dim})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features from video windows"
    )
    parser.add_argument("--labels-file", required=True, help="Path to labels.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for features")
    parser.add_argument("--backbone", default="dinov2-base", help="DINOv2 model variant")
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument(
        "--device", default="auto",
        help="Device for DINOv2 inference (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--pool-mode", default="mean_patch",
        choices=["cls", "mean_patch"],
        help="Feature pooling mode: cls (CLS token) or mean_patch (mean of patch tokens)",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    extract_features(
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        backbone=args.backbone,
        sample_fps=args.sample_fps,
        window_seconds=args.window_seconds,
        frame_size=args.frame_size,
        device=device,
        pool_mode=args.pool_mode,
    )


if __name__ == "__main__":
    main()

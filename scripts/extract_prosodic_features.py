#!/usr/bin/env python3
"""Extract eGeMAPS prosodic features (low-level descriptors, ~100Hz frame rate).

Drop-in alternative to WavLM features for the multimodal pipeline. Each 1-second
audio clip yields a (~100, 25) tensor of prosodic LLDs: f0 (pitch), jitter,
shimmer, HNR, spectral characteristics, MFCC 1-4, formants, etc.

Why prosodic instead of WavLM-base-plus
---------------------------------------
D1b analysis showed audio_repr is 92-98% linearly predictable from video_repr
when trained on WavLM-base-plus features (R²(A→V) ≈ 0.95). The two encode
redundant scene-level content — WavLM's phonetic / semantic objectives produce
embeddings that overlap with what DINOv2 captures visually.

eGeMAPS LLDs explicitly target prosodic dimensions (pitch contour, voice
quality, energy dynamics) that are mechanistically harder to predict from
visible behavior. They're the feature set traditionally used in synchrony /
affect research. If WavLM-vs-video redundancy is feature-property rather than
training-dynamics (D3 finding), switching to LLDs should break the redundancy
and reveal whether real audio-side synchrony signal exists at this dataset
size.

Usage:
    python scripts/extract_prosodic_features.py \\
        --labels-file data/labels.csv \\
        --output-dir data/prosodic_features \\
        --chunk-duration 1.0

Dry-run first (recommended, per dry-run discipline memory):
    python scripts/extract_prosodic_features.py \\
        --labels-file data/labels.csv \\
        --output-dir data/prosodic_features_dryrun \\
        --limit 100

Output matches the WavLM extraction format so the multimodal training script
can use it without changes:
    data/prosodic_features/
        feature_index.csv
        features/
            {stem}_{second}_{hash}.pt  # (n_frames, 25) float32 tensor
"""

import argparse
import csv
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def feature_filename(video_path: str, second: int) -> str:
    """Same hashing scheme as extract_audio_features.py so file names collide
    across feature dirs — makes joining trivial."""
    key = f"{video_path}:{second}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    video_stem = Path(video_path).stem[:20]
    return f"{video_stem}_{second:05d}_{h}.pt"


def load_labels(labels_file: str) -> list[dict]:
    entries = []
    with open(labels_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "video_path": row["video_path"],
                "second": int(row["second"]),
                "label": float(row["label"]),
                "subject_id": row.get("subject_id", ""),
                "session": row.get("session", ""),
            })
    logger.info(f"Loaded {len(entries)} labeled seconds from {labels_file}")
    return entries


def extract_features(
    labels_file: str,
    output_dir: str,
    chunk_duration: float = 1.0,
    sample_rate: int = 16000,
    limit: int | None = None,
) -> None:
    """Extract eGeMAPSv02 LLDs for all labeled audio seconds.

    Mirrors the audio-cache + lenient-boundary patterns from extract_audio_features.py
    so per-file pace stays fast (~50ms per entry after first decode per video).
    """
    try:
        import opensmile
    except ImportError as e:
        raise ImportError(
            "opensmile not installed. Install via:\n"
            f"  $SYNCHRONAI_DIR/ml-env/bin/pip install opensmile\n"
            f"(original error: {e})"
        )

    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # eGeMAPSv02 LLDs: the de facto standard prosodic feature set in
    # affective computing. ~100Hz frame rate. 25 dims per frame.
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    logger.info(f"openSMILE: feature_set=eGeMAPSv02 (LLDs), feature_names={smile.feature_names}")

    entries = load_labels(labels_file)
    if limit is not None and limit > 0:
        entries = entries[:limit]
        logger.info(f"DRY-RUN: limiting to first {limit} entries")

    # Per-video audio cache — load_audio() reads the full decoded WAV from
    # NFS (~16-50 MB), so caching avoids re-reading per second.
    from synchronai.data.audio.processing import load_audio
    cached_video_path: str | None = None
    cached_audio: np.ndarray | None = None
    n_chunk_samples = int(chunk_duration * sample_rate)

    index_rows = []
    n_success = 0
    n_fail = 0
    logged_errors: set = set()
    start_time = time.time()

    feature_dim_observed = None  # set on first successful extract

    for i, entry in enumerate(tqdm(entries, desc="Extracting prosodic features")):
        video_path = entry["video_path"]
        second = entry["second"]

        fname = feature_filename(video_path, second)
        feat_path = features_dir / fname

        # Skip if already extracted
        if feat_path.exists():
            feat = torch.load(feat_path, map_location="cpu", weights_only=True)
            index_rows.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": entry["label"],
                "subject_id": entry["subject_id"],
                "session": entry["session"],
                "feature_dim": int(feat.shape[-1]),
                "n_frames": int(feat.shape[0]),
                "n_layers": 0,
                "all_layers": False,
            })
            n_success += 1
            continue

        # Audio: cache-aware load
        if video_path != cached_video_path:
            try:
                cached_audio = load_audio(video_path, sample_rate)
                cached_video_path = video_path
            except Exception as e:
                if video_path not in logged_errors:
                    logged_errors.add(video_path)
                    logger.warning(f"Failed to load audio from {video_path}: {e}")
                cached_video_path = None
                cached_audio = None
                n_fail += 1
                continue

        if cached_audio is None:
            n_fail += 1
            continue

        # Lenient boundary: pad with zeros if the clip runs past end-of-audio
        # (matches extract_audio_features.py's post-fix behavior).
        start_sample = int(second * sample_rate)
        if start_sample >= len(cached_audio):
            audio = np.zeros(n_chunk_samples, dtype=np.float32)
        else:
            audio = cached_audio[start_sample:start_sample + n_chunk_samples]
            if len(audio) < n_chunk_samples:
                audio = np.pad(audio, (0, n_chunk_samples - len(audio)))

        # opensmile.process_signal expects 1-D or 2-D float array.
        try:
            df = smile.process_signal(audio, sample_rate)
            feat_np = df.values.astype(np.float32)  # (n_frames, n_features)
        except Exception as e:
            if video_path not in logged_errors:
                logged_errors.add(video_path)
                logger.warning(f"opensmile failed for {video_path}@{second}s: {e}")
            n_fail += 1
            continue

        if feature_dim_observed is None:
            feature_dim_observed = feat_np.shape[1]
            logger.info(
                f"First successful extract: shape ({feat_np.shape[0]}, {feat_np.shape[1]})"
            )

        feat = torch.from_numpy(feat_np)
        torch.save(feat, feat_path)

        index_rows.append({
            "feature_file": fname,
            "video_path": video_path,
            "second": second,
            "label": entry["label"],
            "subject_id": entry["subject_id"],
            "session": entry["session"],
            "feature_dim": int(feat.shape[-1]),
            "n_frames": int(feat.shape[0]),
            "n_layers": 0,
            "all_layers": False,
        })
        n_success += 1

        # Periodic index flush (every 5000 entries) so a crash mid-run still
        # leaves a partial index recoverable. Final flush below.
        if (i + 1) % 5000 == 0:
            pd.DataFrame(index_rows).to_csv(output_dir / "feature_index.csv", index=False)

    pd.DataFrame(index_rows).to_csv(output_dir / "feature_index.csv", index=False)
    elapsed = time.time() - start_time
    logger.info(f"Extraction complete in {elapsed:.1f}s")
    logger.info(f"  Success: {n_success}, Failed: {n_fail}")
    logger.info(f"  Features saved to: {features_dir}")
    logger.info(f"  Index saved to: {output_dir / 'feature_index.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N entries (dry-run/smoke test).",
    )
    args = parser.parse_args()

    extract_features(
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract WavLM (or Whisper) audio features and save to disk.

Pre-extracts frozen encoder features for all labeled audio seconds in a
labels.csv file. Pre-extracted features enable fast CPU-based training of
the classification head without needing the encoder loaded during training.

Usage:
    python scripts/extract_audio_features.py \
        --labels-file data/labels.csv \
        --output-dir data/wavlm_features \
        --encoder wavlm-base-plus \
        --device auto

Output:
    data/wavlm_features/
        feature_index.csv      # Maps feature files to metadata
        features/              # Directory of .pt feature tensors
            {stem}_{second}_{hash}.pt  # Each: (n_content_frames, encoder_dim)
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def load_labels(labels_file: str) -> list[dict]:
    """Load labels CSV and return list of dicts."""
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


def extract_audio_chunk(video_path: str, second: int, duration: float = 1.0,
                        sample_rate: int = 16000,
                        _logged_errors: set | None = None) -> np.ndarray | None:
    """Extract audio chunk from video file.

    Returns:
        Audio waveform as float32 numpy array (n_samples,), or None on failure.
    """
    try:
        from synchronai.data.audio.processing import load_audio_chunk
        audio = load_audio_chunk(video_path, second, duration, sample_rate)
        if audio is not None and len(audio) > 0:
            return audio
    except Exception as e:
        # Log the first failure per unique video at WARNING level so it's
        # visible in bsub logs (not buried at DEBUG level).
        if _logged_errors is not None:
            if video_path not in _logged_errors:
                _logged_errors.add(video_path)
                logger.warning(
                    f"Failed to extract audio from {video_path}@{second}s: {e}"
                )
        else:
            logger.warning(
                f"Failed to extract audio from {video_path}@{second}s: {e}"
            )
    return None


def build_encoder(encoder_name: str, device: str):
    """Build the appropriate audio encoder.

    Args:
        encoder_name: "wavlm-base-plus", "wavlm-large", "large-v3", etc.
        device: torch device string

    Returns:
        Tuple of (encoder, encoder_dim, encoder_type)
    """
    if encoder_name.startswith("wavlm") or encoder_name.startswith("microsoft/wavlm"):
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures
        encoder = WavLMEncoderFeatures(
            model_name=encoder_name,
            freeze=True,
        )
        encoder_dim = encoder.encoder_dim
        encoder_type = "wavlm"
    else:
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures
        encoder = WhisperEncoderFeatures(
            model_size=encoder_name,
            freeze=True,
        )
        encoder_dim = encoder.encoder_dim
        encoder_type = "whisper"

    encoder.to(device)
    logger.info(
        f"Built {encoder_type} encoder: {encoder_name}, "
        f"dim={encoder_dim}, device={device}"
    )
    return encoder, encoder_dim, encoder_type


def extract_features(
    labels_file: str,
    output_dir: str,
    encoder_name: str = "wavlm-base-plus",
    chunk_duration: float = 1.0,
    sample_rate: int = 16000,
    device: str = "cpu",
    save_all_layers: bool = False,
) -> None:
    """Extract audio encoder features for all labeled seconds.

    Args:
        labels_file: Path to labels.csv
        output_dir: Directory to save features
        encoder_name: Encoder model name
        chunk_duration: Audio chunk duration in seconds
        sample_rate: Audio sample rate
        device: Device for encoder inference
        save_all_layers: If True, save per-layer hidden states
            (num_layers+1, n_frames, dim) instead of blended (n_frames, dim).
            Enables learnable layer weighting during training.
    """
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    entries = load_labels(labels_file)

    # Build encoder
    encoder, encoder_dim, encoder_type = build_encoder(encoder_name, device)

    # Extract features
    index_rows = []
    n_success = 0
    n_fail = 0
    logged_errors: set = set()  # Track which video paths have logged errors
    start_time = time.time()

    for i, entry in enumerate(tqdm(entries, desc="Extracting audio features")):
        video_path = entry["video_path"]
        second = entry["second"]

        # Generate output filename
        fname = feature_filename(video_path, second)
        feat_path = features_dir / fname

        # Skip if already extracted
        if feat_path.exists():
            # Load to get shape for index
            feat = torch.load(feat_path, map_location="cpu", weights_only=True)
            is_multilayer = feat.ndim == 3 and save_all_layers
            index_rows.append({
                "feature_file": fname,
                "video_path": video_path,
                "second": second,
                "label": entry["label"],
                "subject_id": entry["subject_id"],
                "session": entry["session"],
                "feature_dim": feat.shape[-1],
                "n_frames": feat.shape[-2] if is_multilayer else feat.shape[0],
                "n_layers": feat.shape[0] if is_multilayer else 0,
                "all_layers": is_multilayer,
            })
            n_success += 1
            continue

        # Extract audio
        audio = extract_audio_chunk(
            video_path, second, chunk_duration, sample_rate,
            _logged_errors=logged_errors,
        )
        if audio is None:
            n_fail += 1
            continue

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, n_samples)
        audio_tensor = audio_tensor.to(device)

        # Extract features
        with torch.no_grad():
            if save_all_layers and encoder_type == "wavlm":
                # Save per-layer hidden states: (num_layers+1, n_frames, dim)
                features = encoder.extract_all_layers(
                    audio_tensor, chunk_duration=chunk_duration
                )
                # features: (1, num_layers+1, n_frames, encoder_dim)
                feat = features.squeeze(0).cpu()
                # feat shape: (num_layers+1, n_frames, encoder_dim)
            else:
                features = encoder.extract_features(
                    audio_tensor, pool="none", chunk_duration=chunk_duration
                )
                # features: (1, n_frames, encoder_dim)

                # For Whisper: trim to content frames (avoid silence padding)
                if encoder_type == "whisper":
                    total_frames = features.shape[1]
                    content_frames = max(1, int(total_frames * chunk_duration / 30.0))
                    features = features[:, :content_frames, :]

                feat = features.squeeze(0).cpu()
                # feat shape: (n_frames, encoder_dim)

        torch.save(feat, feat_path)

        n_layers = feat.shape[0] if save_all_layers else 0
        n_frames = feat.shape[-2] if save_all_layers else feat.shape[0]
        feature_dim = feat.shape[-1]

        index_rows.append({
            "feature_file": fname,
            "video_path": video_path,
            "second": second,
            "label": entry["label"],
            "subject_id": entry["subject_id"],
            "session": entry["session"],
            "feature_dim": feature_dim,
            "n_frames": n_frames,
            "n_layers": n_layers,
            "all_layers": save_all_layers,
        })
        n_success += 1

        # Progress logging
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            rate = n_success / elapsed
            logger.info(
                f"  {n_success}/{len(entries)} extracted "
                f"({rate:.1f}/sec, {n_fail} failures)"
            )

    elapsed = time.time() - start_time

    # Write index CSV
    index_path = output_dir / "feature_index.csv"
    if index_rows:
        fieldnames = list(index_rows[0].keys())
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_rows)

    logger.info(f"Extraction complete in {elapsed:.1f}s")
    logger.info(f"  Success: {n_success}, Failed: {n_fail}")
    logger.info(f"  Features saved to: {features_dir}")
    logger.info(f"  Index saved to: {index_path}")
    if index_rows:
        row0 = index_rows[0]
        if row0.get("all_layers"):
            logger.info(
                f"  Feature shape: ({row0['n_layers']}, {row0['n_frames']}, "
                f"{row0['feature_dim']}) [per-layer mode]"
            )
        else:
            logger.info(
                f"  Feature shape: ({row0['n_frames']}, {row0['feature_dim']})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio encoder features from video files"
    )
    parser.add_argument(
        "--labels-file", required=True,
        help="Path to labels.csv with video_path, second, label columns",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for features and index",
    )
    parser.add_argument(
        "--encoder", default="wavlm-base-plus",
        help="Encoder model: wavlm-base-plus, wavlm-large, large-v3, etc.",
    )
    parser.add_argument(
        "--chunk-duration", type=float, default=1.0,
        help="Audio chunk duration in seconds",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Audio sample rate in Hz",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: cpu, cuda, or auto",
    )
    parser.add_argument(
        "--save-all-layers", action="store_true",
        help="Save per-layer hidden states (num_layers+1, n_frames, dim) "
             "instead of blended (n_frames, dim). Enables learnable layer "
             "weighting during training. ~12x more storage.",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    extract_features(
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        encoder_name=args.encoder,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate,
        device=device,
        save_all_layers=args.save_all_layers,
    )


if __name__ == "__main__":
    main()

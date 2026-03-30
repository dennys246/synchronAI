#!/usr/bin/env python3
"""
Extract fNIRS encoder features from trained DDPM U-Net.

Uses the PyTorch encoder replica (unet_encoder_pt.py) with converted weights
to extract bottleneck features from fNIRS recordings. Supports two modes:

1. Per-recording features (for child/adult classification):
   Saves full bottleneck (59, 512) per recording.

2. Multi-scale features (optional):
   Concatenates all encoder levels for richer representations.

Feature extraction follows the same pattern as extract_audio_features.py:
- Resume support (skip existing .pt files)
- feature_index.csv with standard columns
- Deterministic filenames via MD5 hash

Usage:
    python scripts/extract_fnirs_features.py \
        --encoder-weights runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt \
        --data-dirs "/path/to/CARE/NIRS_data:/path/to/PCAT/NIRS_data" \
        --output-dir data/fnirs_encoder_features \
        --multiscale
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def feature_filename(fnirs_path: str, segment_idx: int = 0) -> str:
    """Deterministic filename for a feature file."""
    key = f"{fnirs_path}:{segment_idx}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    stem = Path(fnirs_path).stem[:30]
    return f"{stem}_{segment_idx:03d}_{h}.pt"


def extract_subject_id(path: str) -> str:
    """Extract subject ID from fNIRS file path.

    Matches the logic in training/diffusion/train.py:_extract_subject_id().
    """
    p = Path(path)
    if p.is_dir():
        return p.name
    return p.stem.replace("_Deconvolved", "").replace("_deconvolved", "")


def detect_participant_type(path: str) -> str:
    """Detect child vs adult from fNIRS recording path.

    Confirmed naming conventions by study:

    CARE (.../CARE/NIRS_data/{family}/V0/):
        50001_V0_fNIRS/  → child (5-digit subject ID, last digit = child #)
        5000_V0_fNIRS/   → parent (4-digit subject ID)

    P-CAT R56 (.../R56/NIRS_data/{family}/{family}_DB-DOS/):
        1102-C_fNIRS_DB-DOS/  → child  ("-C_" in directory name)
        1102-P_fNIRS_DB-DOS/  → parent ("-P_" in directory name)

    P-CAT R01 WUSTL (.../WUSTL_data/{timepoint}/nirs_data/dbdos/{family}/):
        11001_C/  → child  ("_C" suffix on recording parent dir)
        11001_P/  → parent ("_P" suffix)

    P-CAT R01 PSU (.../PSU_data/{timepoint}/nirs_data/dbdos/{family}/):
        12001_C/  → child  ("_C" suffix on recording parent dir)
        12001_P/  → parent ("_P" suffix)

    Returns 'child', 'adult', or 'unknown'.
    Override with --participant-labels CSV for custom mappings.
    """
    parts = Path(path).parts

    # Check all path components for child/parent indicators.
    # We scan the full path because the indicator can be on the recording
    # directory itself or on a parent directory.
    for part in parts:
        part_upper = part.upper()

        # --- P-CAT R56 convention: "-C_" or "-P_" in directory name ---
        # e.g., "1102-C_fNIRS_DB-DOS" or "1102-P_fNIRS_DB-DOS"
        if "-C_" in part_upper or part_upper.endswith("-C"):
            return "child"
        if "-P_" in part_upper or part_upper.endswith("-P"):
            return "adult"

        # --- P-CAT R01 convention: "_C" or "_P" suffix ---
        # e.g., "11001_C" or "12001_P"
        # Must be careful not to match other underscores — check it's at the end
        # or followed by a non-alphanumeric
        if part_upper.endswith("_C") and len(part) > 2:
            return "child"
        if part_upper.endswith("_P") and len(part) > 2:
            return "adult"

    # --- CARE convention: subject ID in directory name ---
    # Recording dirs like "50001_V0_fNIRS" (child, 5-digit) or
    # "5000_V0_fNIRS" (parent, 4-digit).
    # The recording directory name (for NIRx) or filename stem (for .snirf)
    # starts with the subject ID.
    p = Path(path)
    name = p.name if p.is_dir() else p.stem

    # Extract leading digits from name
    digits = ""
    for ch in name:
        if ch.isdigit():
            digits += ch
        else:
            break

    if digits:
        if len(digits) == 5 and digits.startswith("5"):
            return "child"
        elif len(digits) == 4 and digits.startswith("5"):
            return "adult"

    return "unknown"


def discover_fnirs_paths(data_dirs: str, signal_type: str = "hemodynamic") -> list[str]:
    """Discover fNIRS recording paths from colon-separated directories.

    Matches LazyFnirsDiscovery logic from training/diffusion/train.py.

    Supported formats (in priority order):
    1. NIRx directories (contain .hdr file) — CARE, P-CAT R56, P-CAT R01
    2. .snirf files (standalone) — CARE (some recordings)
    3. .fif files — preprocessed outputs

    If a directory is a NIRx recording (has .hdr), .snirf/.fif files inside
    it are NOT separately discovered (avoids double-counting).
    """
    paths = []
    nirx_dirs = set()  # Track NIRx dirs to avoid double-counting

    for dir_path in data_dirs.split(":"):
        dir_path = dir_path.strip()
        if not dir_path or not Path(dir_path).exists():
            logger.warning(f"Skipping missing directory: {dir_path}")
            continue

        for root, dirs, files in sorted(Path(dir_path).walk()):
            # Check for NIRx directory (has .hdr file)
            if any(f.endswith(".hdr") for f in files):
                paths.append(str(root))
                nirx_dirs.add(str(root))
                continue

            # Skip .snirf/.fif files inside NIRx directories
            if str(root) in nirx_dirs:
                continue

            for fname in sorted(files):
                if fname.endswith(".snirf") or fname.endswith(".fif"):
                    is_deconvolved = "_Deconvolved" in fname or "deconvolved" in fname.lower()
                    if signal_type == "hemodynamic" and is_deconvolved:
                        continue
                    if signal_type == "neural" and not is_deconvolved:
                        continue
                    paths.append(str(root / fname))

    logger.info(f"Discovered {len(paths)} fNIRS recordings from {data_dirs}")
    return paths


def load_and_normalize_recording(
    fnirs_path: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_pairs: list[str],
    hb_types: list[str],
    target_sfreq: Optional[float] = None,
    duration_seconds: float = 60.0,
) -> Optional[np.ndarray]:
    """Load an fNIRS recording, preprocess, and normalize.

    Returns:
        Normalized array of shape (time, features) or None on failure.
    """
    try:
        from synchronai.data.fnirs.processing import extract_hemoglobin_pairs, load_fnirs
        from synchronai.data.fnirs.dataset import _align_pairs

        raw = load_fnirs(fnirs_path, deconvolution=False)
        if target_sfreq is not None:
            raw = raw.copy().resample(target_sfreq)

        x, meta = extract_hemoglobin_pairs(raw)

        # Align to target pair set
        x = _align_pairs(x, meta, target_pairs)

        # Reshape to (time, features): (time, pairs, hb_types) -> (time, pairs*hb_types)
        n_time, n_pairs, n_hb = x.shape
        x = x.reshape(n_time, n_pairs * n_hb).astype(np.float32)

        # Normalize using training statistics
        std_safe = np.maximum(feature_std, 1e-8)
        x = (x - feature_mean) / std_safe
        x = np.clip(x, -6.0, 6.0)

        return x

    except Exception as e:
        logger.warning(f"Failed to load {fnirs_path}: {e}")
        return None


def window_recording(
    x: np.ndarray,
    model_len: int,
    stride_seconds: float = 60.0,
    sfreq_hz: float = 7.8125,
) -> list[np.ndarray]:
    """Slice a recording into fixed-length windows.

    Args:
        x: (total_time, features) normalized recording
        model_len: Expected input length for encoder (e.g., 472)
        stride_seconds: Stride between windows in seconds.
            60.0 = non-overlapping (for 60s model_len).
            30.0 = 50% overlap.
        sfreq_hz: Sampling frequency

    Returns:
        List of (model_len, features) arrays
    """
    stride_samples = int(stride_seconds * sfreq_hz)
    if stride_samples < 1:
        stride_samples = model_len

    windows = []
    start = 0
    while start + model_len <= x.shape[0]:
        windows.append(x[start : start + model_len])
        start += stride_samples

    # If recording is shorter than model_len, pad and use as single window
    if not windows and x.shape[0] > 0:
        padded = np.zeros((model_len, x.shape[1]), dtype=np.float32)
        padded[: x.shape[0]] = x
        windows.append(padded)

    return windows


def extract_features(
    data_dirs: str,
    output_dir: str,
    encoder_weights_path: str,
    multiscale: bool = False,
    participant_labels: Optional[str] = None,
    signal_type: str = "hemodynamic",
    device: str = "cpu",
    stride_seconds: float = 60.0,
    random_init: bool = False,
) -> None:
    """Extract fNIRS encoder features for all discovered recordings."""

    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    logger.info(f"Loading PyTorch encoder from {encoder_weights_path}")
    save_data = torch.load(encoder_weights_path, map_location=device, weights_only=False)
    encoder_config = save_data["encoder_config"]
    diffusion_config = save_data["config"]

    from synchronai.models.fnirs.unet_encoder_pt import FnirsUNetEncoderPT

    encoder = FnirsUNetEncoderPT(
        input_length=encoder_config["input_length"],
        feature_dim=encoder_config["feature_dim"],
        base_width=encoder_config["base_width"],
        depth=encoder_config["depth"],
        time_embed_dim=encoder_config["time_embed_dim"],
        dropout=encoder_config["dropout"],
    )
    if random_init:
        logger.info("ABLATION MODE: Using randomly initialized encoder (no pretrained weights)")
    else:
        encoder.load_state_dict(save_data["state_dict"])
    encoder.eval()
    encoder.to(device)

    feature_dim = encoder_config["multiscale_dim"] if multiscale else encoder_config["bottleneck_dim"]
    model_len = encoder_config["input_length"]

    logger.info(
        f"Encoder loaded: bottleneck_dim={encoder_config['bottleneck_dim']}, "
        f"multiscale_dim={encoder_config['multiscale_dim']}, "
        f"using={'multiscale' if multiscale else 'bottleneck'} (dim={feature_dim})"
    )

    # Load normalization stats from config
    feature_mean = np.array(diffusion_config["feature_mean"], dtype=np.float32)
    feature_std = np.array(diffusion_config["feature_std"], dtype=np.float32)
    target_pairs = diffusion_config["pair_names"]
    hb_types = diffusion_config["hb_types"]
    duration_seconds = diffusion_config["duration_seconds"]
    target_sfreq = diffusion_config["sfreq_hz"]

    # Load participant labels if provided
    participant_map = {}
    if participant_labels and Path(participant_labels).exists():
        import pandas as pd
        df = pd.read_csv(participant_labels)
        for _, row in df.iterrows():
            participant_map[row["fnirs_path"]] = row["participant_type"]
        logger.info(f"Loaded {len(participant_map)} participant labels from {participant_labels}")

    # Discover recordings
    fnirs_paths = discover_fnirs_paths(data_dirs, signal_type)
    if not fnirs_paths:
        logger.error("No fNIRS recordings found!")
        return

    # Extract features
    index_rows = []
    n_success = 0
    n_fail = 0
    start_time = time.time()

    for i, fnirs_path in enumerate(tqdm(fnirs_paths, desc="Extracting fNIRS features")):
        subject_id = extract_subject_id(fnirs_path)
        participant_type = participant_map.get(fnirs_path, detect_participant_type(fnirs_path))

        # Check if first window already extracted (resume support)
        first_fname = feature_filename(fnirs_path, segment_idx=0)
        first_feat_path = features_dir / first_fname
        if first_feat_path.exists():
            # Reload all windows for this recording
            seg_idx = 0
            while True:
                fname = feature_filename(fnirs_path, segment_idx=seg_idx)
                feat_path = features_dir / fname
                if not feat_path.exists():
                    break
                feat = torch.load(feat_path, map_location="cpu", weights_only=True)
                index_rows.append({
                    "feature_file": fname,
                    "fnirs_path": fnirs_path,
                    "subject_id": subject_id,
                    "participant_type": participant_type,
                    "window_idx": seg_idx,
                    "feature_dim": feat.shape[-1],
                    "n_frames": feat.shape[0] if feat.ndim == 2 else 1,
                    "multiscale": multiscale,
                })
                seg_idx += 1
                n_success += 1
            continue

        # Load and normalize full recording
        x = load_and_normalize_recording(
            fnirs_path,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_pairs=target_pairs,
            hb_types=hb_types,
            target_sfreq=target_sfreq,
            duration_seconds=duration_seconds,
        )
        if x is None:
            n_fail += 1
            continue

        # Slice into windows
        windows = window_recording(x, model_len, stride_seconds, target_sfreq)

        for seg_idx, window in enumerate(windows):
            fname = feature_filename(fnirs_path, segment_idx=seg_idx)
            feat_path = features_dir / fname

            x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                if multiscale:
                    outputs = encoder(x_tensor, return_all_levels=True)
                    pooled_parts = []
                    for key in sorted(outputs.keys()):
                        pooled_parts.append(outputs[key].squeeze(0).mean(dim=0))
                    feat_to_save = torch.cat(pooled_parts).cpu()  # (960,)
                else:
                    features = encoder(x_tensor)  # (1, 59, 512)
                    feat_to_save = features.squeeze(0).cpu()  # (59, 512)

            torch.save(feat_to_save, feat_path)

            index_rows.append({
                "feature_file": fname,
                "fnirs_path": fnirs_path,
                "subject_id": subject_id,
                "participant_type": participant_type,
                "window_idx": seg_idx,
                "feature_dim": feat_to_save.shape[-1] if feat_to_save.ndim > 1 else feat_to_save.shape[0],
                "n_frames": feat_to_save.shape[0] if feat_to_save.ndim == 2 else 1,
                "multiscale": multiscale,
            })
            n_success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = n_success / elapsed if elapsed > 0 else 0
            logger.info(f"  {n_success} windows from {i+1}/{len(fnirs_paths)} recordings ({rate:.1f}/sec, {n_fail} failures)")

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
        logger.info(f"  Feature dim: {row0['feature_dim']}, frames: {row0['n_frames']}")
        # Report participant type distribution
        types = [r["participant_type"] for r in index_rows]
        for t in sorted(set(types)):
            logger.info(f"  {t}: {types.count(t)} windows")
        n_unknown = types.count("unknown")
        if n_unknown > len(types) * 0.5:
            logger.warning(
                f"  WARNING: {n_unknown}/{len(types)} windows have unknown "
                f"participant_type! Check detect_participant_type() for these paths."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract fNIRS encoder features from trained DDPM U-Net"
    )
    parser.add_argument(
        "--encoder-weights", required=True,
        help="Path to converted PyTorch encoder weights (.pt file from convert_fnirs_tf_to_pt.py)",
    )
    parser.add_argument(
        "--data-dirs", required=True,
        help="Colon-separated fNIRS data directories (same format as generative_pretrain.sh)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for features and index",
    )
    parser.add_argument(
        "--multiscale", action="store_true",
        help="Extract multi-scale features (all encoder levels concatenated, "
             "960-dim for depth=3) instead of bottleneck-only (512-dim)",
    )
    parser.add_argument(
        "--participant-labels", default=None,
        help="Optional CSV with fnirs_path,participant_type columns for child/adult labels",
    )
    parser.add_argument(
        "--signal-type", default="hemodynamic",
        choices=["hemodynamic", "neural"],
        help="Signal type to extract (hemodynamic=default, neural=deconvolved)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for encoder inference (cpu recommended, model is small)",
    )
    parser.add_argument(
        "--stride-seconds", type=float, default=60.0,
        help="Stride between windows in seconds. 60.0=non-overlapping (default), "
             "30.0=50%% overlap for more samples from each recording.",
    )
    parser.add_argument(
        "--random-init", action="store_true",
        help="Use randomly initialized encoder (no pretrained weights). "
             "For ablation: proves pretraining is necessary.",
    )

    args = parser.parse_args()

    extract_features(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        encoder_weights_path=args.encoder_weights,
        multiscale=args.multiscale,
        participant_labels=args.participant_labels,
        signal_type=args.signal_type,
        device=args.device,
        stride_seconds=args.stride_seconds,
        random_init=args.random_init,
    )


if __name__ == "__main__":
    main()

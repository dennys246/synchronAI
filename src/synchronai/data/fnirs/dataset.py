"""
fNIRS training dataset utilities.

This module converts preprocessed hemoglobin time series into fixed-length
windows suitable for training a 1D diffusion model.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from synchronai.data.fnirs.processing import HemoglobinMeta, extract_hemoglobin_pairs, load_fnirs
from synchronai.utils.logging import get_logger


@dataclass(frozen=True)
class FnirsTrainingData:
    windows: np.ndarray  # (n_windows, time, features)
    sfreq_hz: float
    duration_seconds: float
    pair_names: List[str]
    hb_types: List[str]
    feature_mean: np.ndarray  # (features,)
    feature_std: np.ndarray  # (features,)


def _align_pairs(
    x: np.ndarray, meta: HemoglobinMeta, target_pair_names: List[str]
) -> np.ndarray:
    """
    Align (time, pairs, hb) -> (time, target_pairs, hb) by name, filling missing
    pairs with zeros (keeps feature dimensionality stable across recordings).
    """
    pair_to_idx = {name: i for i, name in enumerate(meta.pair_names)}
    time, _, hb = x.shape
    aligned = np.zeros((time, len(target_pair_names), hb), dtype=np.float32)
    for target_idx, pair_name in enumerate(target_pair_names):
        src_idx = pair_to_idx.get(pair_name)
        if src_idx is None:
            continue
        aligned[:, target_idx, :] = x[:, src_idx, :]
    return aligned


def _standardize_windows(windows: np.ndarray, eps: float = 1e-6):
    """
    Feature-wise standardization over (window,time) axes.
    """
    # windows: (n, time, features)
    mean = windows.mean(axis=(0, 1))
    var = windows.var(axis=(0, 1))
    std = np.sqrt(np.maximum(var, eps))
    normed = (windows - mean[None, None, :]) / std[None, None, :]
    normed = np.clip(normed, -6.0, 6.0).astype(np.float32)
    return normed, mean.astype(np.float32), std.astype(np.float32)


def standardize_with_stats(
    windows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Standardize windows using pre-computed mean and std.

    Args:
        windows: Array of shape (n_windows, time, features)
        mean: Feature means of shape (features,)
        std: Feature stds of shape (features,)
        eps: Small value to prevent division by zero

    Returns:
        Normalized windows clipped to [-6, 6]
    """
    std_safe = np.maximum(std, eps)
    normed = (windows - mean[None, None, :]) / std_safe[None, None, :]
    normed = np.clip(normed, -6.0, 6.0).astype(np.float32)
    return normed


def load_training_windows(
    fnirs_paths: List[str],
    *,
    duration_seconds: float = 120.0,
    target_sfreq_hz: Optional[float] = None,
    segments_per_recording: int = 8,
    seed: int = 0,
    deconvolution: bool = False,
    max_recordings: Optional[int] = None,
    normalize: bool = True,
    external_mean: Optional[np.ndarray] = None,
    external_std: Optional[np.ndarray] = None,
) -> FnirsTrainingData:
    """
    Load recordings, preprocess via HRfunc, and return standardized windows.

    Args:
        fnirs_paths: List of paths to fNIRS recordings
        duration_seconds: Duration of each window in seconds
        target_sfreq_hz: Target sampling frequency (None to use original)
        segments_per_recording: Number of random segments to extract per recording
        seed: Random seed for reproducibility
        deconvolution: Whether to use deconvolution in preprocessing
        max_recordings: Maximum number of recordings to load
        normalize: Whether to normalize the windows (default True)
        external_mean: Optional pre-computed mean for normalization (requires normalize=True)
        external_std: Optional pre-computed std for normalization (requires normalize=True)

    Returns:
        FnirsTrainingData with windows and normalization statistics
    """
    logger = get_logger(__name__)
    if not fnirs_paths:
        raise ValueError("No fNIRS paths provided.")

    rng = np.random.default_rng(seed)
    paths = list(fnirs_paths[: max_recordings or len(fnirs_paths)])
    logger.info(
        "Loading fNIRS recordings=%d (received %d paths, max_recordings=%s, segments_per_recording=%d, duration_seconds=%.1f, target_sfreq_hz=%s)",
        len(paths),
        len(fnirs_paths),
        str(max_recordings),
        segments_per_recording,
        duration_seconds,
        str(target_sfreq_hz),
    )

    # First pass: determine target channel set + sampling rate.
    first_raw = load_fnirs(paths[0], deconvolution=deconvolution)
    if target_sfreq_hz is not None:
        first_raw = first_raw.copy().resample(target_sfreq_hz)
    first_x, first_meta = extract_hemoglobin_pairs(first_raw)

    target_pair_names = first_meta.pair_names
    hb_types = first_meta.hb_types
    sfreq_hz = float(first_meta.sfreq_hz if target_sfreq_hz is None else target_sfreq_hz)
    target_len = int(round(duration_seconds * sfreq_hz))
    if target_len <= 1:
        raise ValueError(f"Invalid duration_seconds={duration_seconds} for sfreq={sfreq_hz}.")

    windows: List[np.ndarray] = []

    for idx, path in enumerate(paths):
        logger.debug("Loading fNIRS recording %d/%d: %s", idx + 1, len(paths), path)
        try:
            raw = load_fnirs(path, deconvolution=deconvolution)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Skipping recording {path}: {e}")
            continue

        if target_sfreq_hz is not None:
            raw = raw.copy().resample(target_sfreq_hz)

        x, meta = extract_hemoglobin_pairs(raw)  # (time, pairs, hb)
        if meta.hb_types != hb_types:
            raise ValueError(
                f"Mixed hemoglobin channel types across recordings: {meta.hb_types} vs {hb_types}"
            )
        x = _align_pairs(x, meta, target_pair_names)

        time_len = x.shape[0]
        if time_len < target_len:
            pad = np.zeros((target_len - time_len, x.shape[1], x.shape[2]), dtype=np.float32)
            window = np.concatenate([x, pad], axis=0)
            windows.append(window)
            # Explicit cleanup to prevent memory fragmentation and segfaults
            del raw, x, meta, pad, window
            gc.collect()
            continue

        max_start = time_len - target_len
        # Keep at least 1 segment per recording.
        num_segments = max(1, int(segments_per_recording))
        starts = rng.integers(0, max_start + 1, size=num_segments)
        for start in starts:
            windows.append(x[start : start + target_len])

        # Explicit cleanup after processing each recording to prevent memory fragmentation
        del raw, x, meta
        gc.collect()

    # (n_windows, time, pairs, hb) -> (n_windows, time, features)
    # Validate all windows have the same shape before stacking
    if windows:
        first_shape = windows[0].shape
        for i, win in enumerate(windows):
            if win.shape != first_shape:
                raise ValueError(
                    f"Window shape mismatch: window {i} has shape {win.shape}, "
                    f"expected {first_shape}. This may indicate data processing issues."
                )

    windows_np = np.stack(windows, axis=0)

    # Validate shape before reshaping to prevent segfaults
    if len(windows_np.shape) != 4:
        raise ValueError(
            f"Unexpected windows_np shape: {windows_np.shape}. "
            f"Expected 4D array (n_windows, time, pairs, hb)."
        )

    windows_np = windows_np.reshape(windows_np.shape[0], windows_np.shape[1], -1).astype(np.float32)

    if normalize:
        if external_mean is not None and external_std is not None:
            # Use externally provided statistics (e.g., from running stats)
            windows_norm = standardize_with_stats(windows_np, external_mean, external_std)
            mean = external_mean.astype(np.float32)
            std = external_std.astype(np.float32)
        else:
            # Compute statistics from this batch
            windows_norm, mean, std = _standardize_windows(windows_np)
    else:
        # Return raw (unnormalized) windows - useful for computing running stats
        windows_norm = windows_np
        mean = np.zeros(windows_np.shape[-1], dtype=np.float32)
        std = np.ones(windows_np.shape[-1], dtype=np.float32)

    return FnirsTrainingData(
        windows=windows_norm,
        sfreq_hz=sfreq_hz,
        duration_seconds=duration_seconds,
        pair_names=target_pair_names,
        hb_types=hb_types,
        feature_mean=mean,
        feature_std=std,
    )

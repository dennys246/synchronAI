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

from synchronai.data.fnirs.processing import (
    HemoglobinMeta,
    extract_hemoglobin_pairs,
    load_fnirs,
    read_raw_fnirs,
)
from synchronai.data.fnirs.quality_control import QualityReport, run_quality_control
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
    per_pair: bool = False,
    qc_cache_path: Optional[str] = None,
    # Quality control parameters
    enable_qc: bool = False,
    sci_threshold: float = 0.5,
    snr_threshold: float = 5.0,
    cardiac_band: tuple = (0.8, 1.5),
    cardiac_peak_ratio: float = 2.0,
    require_cardiac: bool = True,
    hrf_band: tuple = (0.01, 0.2),
    peak_power_low: Optional[float] = None,
    peak_power_high: Optional[float] = None,
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
        per_pair: Whether to explode windows into per-pair samples
        enable_qc: Enable the multi-stage quality control pipeline
        sci_threshold: Minimum SCI to keep a channel (0-1). Literature recommends 0.75-0.95
        snr_threshold: Minimum scan-level SNR. Scans below this are rejected
        cardiac_band: Frequency range for cardiac signal in Hz
        cardiac_peak_ratio: Min ratio of cardiac peak to median PSD
        require_cardiac: Reject channels without detectable cardiac signal
        hrf_band: Frequency range for hemodynamic response (signal band for SNR)
        peak_power_low: Min acceptable peak PSD (below = noise-dominated)
        peak_power_high: Max acceptable peak PSD (above = motion artifact)

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

    # Load QC cache if available
    qc_cache: Dict[str, Dict] = {}
    if qc_cache_path:
        from synchronai.data.fnirs.quality_control import load_qc_cache
        qc_cache = load_qc_cache(qc_cache_path)
        logger.info("Loaded QC cache with %d entries from %s", len(qc_cache), qc_cache_path)

    if enable_qc:
        logger.info(
            "QC enabled: sci_threshold=%.2f, snr_threshold=%.1f, require_cardiac=%s, "
            "cardiac_band=%s, hrf_band=%s",
            sci_threshold, snr_threshold, require_cardiac,
            str(cardiac_band), str(hrf_band),
        )

    # First pass: determine target channel set + sampling rate.
    # Try paths until we find one that loads successfully.
    first_meta = None
    for first_path in paths:
        try:
            first_raw = load_fnirs(first_path, deconvolution=deconvolution)
            if target_sfreq_hz is not None:
                first_raw = first_raw.copy().resample(target_sfreq_hz)
            first_x, first_meta = extract_hemoglobin_pairs(first_raw)
            break
        except (ValueError, RuntimeError) as e:
            logger.warning("Skipping %s for config setup: %s", first_path, e)
            continue
    if first_meta is None:
        logger.warning("No valid recordings found in this batch for config setup")
        return FnirsTrainingData(
            windows=np.zeros((0, 0, 0), dtype=np.float32),
            sfreq_hz=target_sfreq_hz or 7.8125,
            duration_seconds=duration_seconds,
            pair_names=[],
            hb_types=["hbo", "hbr"],
            feature_mean=np.zeros(0, dtype=np.float32),
            feature_std=np.ones(0, dtype=np.float32),
        )

    target_pair_names = first_meta.pair_names
    hb_types = first_meta.hb_types
    sfreq_hz = float(first_meta.sfreq_hz if target_sfreq_hz is None else target_sfreq_hz)
    target_len = int(round(duration_seconds * sfreq_hz))
    if target_len <= 1:
        raise ValueError(f"Invalid duration_seconds={duration_seconds} for sfreq={sfreq_hz}.")

    windows: List[np.ndarray] = []
    qc_reports: List[QualityReport] = []
    n_qc_rejected = 0

    for idx, path in enumerate(paths):
        logger.debug("Loading fNIRS recording %d/%d: %s", idx + 1, len(paths), path)

        # --- QC path: read raw first, run checks, then preprocess ---
        if enable_qc:
            # Check QC cache first — skip expensive processing for known-rejected
            cached = qc_cache.get(path)
            if cached is not None:
                if not cached["scan_passed"]:
                    logger.debug(
                        "QC CACHED REJECT %d/%d %s (tier=%s)",
                        idx + 1, len(paths), path, cached["quality_tier"],
                    )
                    n_qc_rejected += 1
                    continue
                # Cached pass — still need to load data, but skip QC
                logger.debug("QC CACHED PASS %d/%d %s", idx + 1, len(paths), path)
                try:
                    raw = load_fnirs(path, deconvolution=deconvolution)
                except (ValueError, RuntimeError) as e:
                    logger.warning("Skipping recording %s (load failed): %s", path, e)
                    continue
            else:
                # No cache entry — run full QC
                try:
                    raw_scan = read_raw_fnirs(path)
                except (ValueError, RuntimeError) as e:
                    logger.warning("Skipping recording %s (read failed): %s", path, e)
                    continue

                try:
                    preprocessed = load_fnirs(path, deconvolution=deconvolution)
                except (ValueError, RuntimeError) as e:
                    logger.warning("Skipping recording %s (preprocess failed): %s", path, e)
                    del raw_scan
                    gc.collect()
                    continue

                qc_report = run_quality_control(
                    raw_scan,
                    preprocessed,
                    sci_threshold=sci_threshold,
                    snr_threshold=snr_threshold,
                    cardiac_band=cardiac_band,
                    cardiac_peak_ratio=cardiac_peak_ratio,
                    require_cardiac=require_cardiac,
                    hrf_band=hrf_band,
                    peak_power_low=peak_power_low,
                    peak_power_high=peak_power_high,
                )
                qc_reports.append(qc_report)
                del raw_scan

                # Save to cache
                if qc_cache_path:
                    from synchronai.data.fnirs.quality_control import save_qc_result
                    save_qc_result(qc_cache_path, path, qc_report)

                if not qc_report.scan_passed:
                    logger.warning(
                        "QC REJECTED recording %d/%d %s: %s",
                        idx + 1, len(paths), path,
                        "; ".join(qc_report.rejection_reasons),
                    )
                    n_qc_rejected += 1
                    del preprocessed
                    gc.collect()
                    continue

                logger.debug(
                    "QC passed recording %d/%d: %d/%d channels kept, SNR=%.2f",
                    idx + 1, len(paths),
                    qc_report.n_channels_after, qc_report.n_channels_before,
                    qc_report.scan_snr or 0.0,
                )

                raw = preprocessed

        # --- Standard path: no QC ---
        else:
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

    if enable_qc:
        logger.info(
            "QC summary: %d/%d recordings passed (%d rejected)",
            len(paths) - n_qc_rejected, len(paths), n_qc_rejected,
        )

    # (n_windows, time, pairs, hb) -> (n_windows, time, features)
    if not windows:
        logger.warning("No valid windows in this batch — all recordings were rejected or empty")
        return FnirsTrainingData(
            windows=np.zeros((0, 0, 0), dtype=np.float32),
            sfreq_hz=target_sfreq_hz or 7.8125,
            duration_seconds=duration_seconds,
            pair_names=[],
            hb_types=["hbo", "hbr"],
            feature_mean=np.zeros(0, dtype=np.float32),
            feature_std=np.ones(0, dtype=np.float32),
        )

    # Validate all windows have the same shape before stacking
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

    if per_pair:
        # Explode (n_windows, time, pairs, hb) → (n_windows * pairs, time, hb)
        # Each pair becomes an independent sample with feature_dim=len(hb_types)
        n_win, t_len, n_pairs, n_hb = windows_np.shape
        # Transpose to (n_windows, pairs, time, hb) then reshape
        windows_np = windows_np.transpose(0, 2, 1, 3)  # (n_win, pairs, time, hb)
        windows_np = windows_np.reshape(n_win * n_pairs, t_len, n_hb).astype(np.float32)
        logger.info(
            f"Per-pair mode: {n_win} windows × {n_pairs} pairs = "
            f"{windows_np.shape[0]} samples, feature_dim={n_hb}"
        )
        # In per-pair mode, pair_names is meaningless (all pairs are pooled)
        target_pair_names = ["any"]
    else:
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

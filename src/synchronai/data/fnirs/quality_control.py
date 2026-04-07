"""
fNIRS signal quality control.

Multi-stage QC pipeline for fNIRS recordings:
  1. Scalp Coupling Index (SCI) — channel-level, on raw optical data
  2. Cardiac band presence — channel-level, on raw optical data
  3. PSD-based SNR — scan-level, on preprocessed hemoglobin data
  4. Peak power bounds — scan-level, on preprocessed hemoglobin data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch

from synchronai.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityTierThresholds:
    """Thresholds defining quality tiers for fNIRS recordings.

    Tiers (highest to lowest):
        gold:        Pristine data — high SCI, high SNR, cardiac in all pairs.
        standard:    Normal QC pass — the current default filtering level.
        salvageable: Noisy but usable — normally rejected, but kept for
                     robustness evaluation.
        rejected:    Too noisy to use in any analysis.
    """

    # Gold tier: pristine, low-motion
    gold_sci: float = 0.90
    gold_snr: float = 10.0
    gold_require_all_cardiac: bool = True

    # Standard tier: normal QC pass (values match current defaults)
    standard_sci: float = 0.75
    standard_snr: float = 5.0

    # Salvageable tier: high-motion, would normally be rejected
    salvageable_sci: float = 0.40
    salvageable_snr: float = 2.0


# Canonical tier names, ordered from best to worst quality.
QUALITY_TIERS = ("gold", "standard", "salvageable", "rejected")


@dataclass
class QualityReport:
    """Results from quality control checks on a single recording."""

    scan_passed: bool
    n_channels_before: int
    n_channels_after: int
    channel_mask: Dict[str, bool]  # pair_name -> passed
    scan_snr: Optional[float] = None
    peak_power: Optional[float] = None
    channel_sci: Dict[str, float] = field(default_factory=dict)
    cardiac_present: Dict[str, bool] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)
    quality_tier: str = "rejected"  # one of QUALITY_TIERS


# ---------------------------------------------------------------------------
# Channel-level checks (run on raw optical data, before bandpass filtering)
# ---------------------------------------------------------------------------


def _get_wavelength_pairs(raw) -> Dict[str, List[int]]:
    """Group raw optical channels by source-detector pair.

    Expects channel names like ``S1_D1 760``, ``S1-D1 850``, etc.
    Returns ``{pair_key: [idx_wavelength1, idx_wavelength2]}``.
    """
    pairs: Dict[str, List[int]] = {}
    for idx, name in enumerate(raw.ch_names):
        # Split off wavelength suffix (last space-separated token)
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            pair_key = parts[0]
            pairs.setdefault(pair_key, []).append(idx)
    return pairs


def compute_scalp_coupling_index(
    raw,
    cardiac_band: Tuple[float, float] = (0.8, 1.5),
) -> Dict[str, float]:
    """Compute SCI for each source-detector pair from raw optical data.

    SCI is the absolute Pearson correlation between the two wavelengths at
    each pair, bandpass-filtered to the cardiac frequency band.  A high SCI
    (close to 1.0) indicates good optode-scalp coupling because the cardiac
    pulsation is visible in both wavelengths.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS recording (optical intensity / CW amplitude).
    cardiac_band : tuple of float
        (low, high) frequency bounds for the cardiac band in Hz.

    Returns
    -------
    dict
        ``{pair_name: sci_value}`` where sci_value is in [0, 1].
    """
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()  # (channels, time)
    pairs = _get_wavelength_pairs(raw)
    nyq = sfreq / 2.0

    lo, hi = cardiac_band
    # Guard against Nyquist violations (low sfreq devices)
    if hi >= nyq:
        logger.warning(
            "Cardiac high freq %.1f Hz >= Nyquist %.1f Hz; clamping.", hi, nyq
        )
        hi = nyq - 0.1
    if lo >= hi:
        logger.warning("Cardiac band invalid after clamping; skipping SCI.")
        return {}

    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")

    sci_values: Dict[str, float] = {}
    for pair_key, indices in pairs.items():
        if len(indices) != 2:
            continue
        try:
            sig1 = filtfilt(b, a, data[indices[0]])
            sig2 = filtfilt(b, a, data[indices[1]])
            corr = float(np.corrcoef(sig1, sig2)[0, 1])
            sci_values[pair_key] = abs(corr)
        except Exception as exc:
            logger.debug("SCI computation failed for %s: %s", pair_key, exc)
            sci_values[pair_key] = 0.0
    return sci_values


def check_cardiac_presence(
    raw,
    cardiac_band: Tuple[float, float] = (0.8, 1.5),
    min_peak_ratio: float = 2.0,
) -> Dict[str, bool]:
    """Check whether a cardiac peak is present in each channel's PSD.

    For each channel we compute the Welch PSD, then compare the peak power
    inside the cardiac band to the median power across all frequencies.
    A ratio above ``min_peak_ratio`` indicates a detectable heartbeat.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS recording (optical intensity).
    cardiac_band : tuple of float
        (low, high) frequency bounds for the cardiac band in Hz.
    min_peak_ratio : float
        Minimum ratio of cardiac-band peak to overall median PSD to count
        as "cardiac present".

    Returns
    -------
    dict
        ``{channel_name: True/False}``.
    """
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()

    results: Dict[str, bool] = {}
    for idx, name in enumerate(raw.ch_names):
        try:
            freqs, psd = welch(data[idx], fs=sfreq, nperseg=min(len(data[idx]), int(4 * sfreq)))
            cardiac_mask = (freqs >= cardiac_band[0]) & (freqs <= cardiac_band[1])
            if not cardiac_mask.any():
                results[name] = False
                continue
            peak_in_band = float(np.max(psd[cardiac_mask]))
            median_psd = float(np.median(psd[psd > 0])) if np.any(psd > 0) else 1.0
            results[name] = (peak_in_band / median_psd) >= min_peak_ratio
        except Exception as exc:
            logger.debug("Cardiac check failed for %s: %s", name, exc)
            results[name] = False
    return results


def _cardiac_by_pair(
    cardiac_per_channel: Dict[str, bool],
) -> Dict[str, bool]:
    """Collapse per-channel cardiac results to per-pair (both wavelengths must pass)."""
    pair_results: Dict[str, List[bool]] = {}
    for ch_name, present in cardiac_per_channel.items():
        parts = ch_name.rsplit(" ", 1)
        pair_key = parts[0] if len(parts) == 2 else ch_name
        pair_results.setdefault(pair_key, []).append(present)
    return {pair: all(vals) for pair, vals in pair_results.items()}


# ---------------------------------------------------------------------------
# Scan-level checks (run on preprocessed hemoglobin data)
# ---------------------------------------------------------------------------


def compute_scan_snr(
    haemo_data: np.ndarray,
    sfreq: float,
    hrf_band: Tuple[float, float] = (0.01, 0.2),
) -> float:
    """Compute PSD-based SNR for a hemoglobin recording.

    Signal is defined as power in the HRF band (default 0.01–0.2 Hz).
    Noise is power outside that band.

    Parameters
    ----------
    haemo_data : np.ndarray
        Hemoglobin time-series, shape ``(channels, time)`` or ``(time,)``.
    sfreq : float
        Sampling frequency in Hz.
    hrf_band : tuple of float
        (low, high) frequency bounds for the HRF / signal band.

    Returns
    -------
    float
        Mean SNR across channels (linear scale, not dB).
    """
    if haemo_data.ndim == 1:
        haemo_data = haemo_data[None, :]

    snrs = []
    for ch_data in haemo_data:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(len(ch_data), int(8 * sfreq)))
        signal_mask = (freqs >= hrf_band[0]) & (freqs <= hrf_band[1])
        noise_mask = ~signal_mask & (freqs > 0)  # exclude DC
        signal_power = float(np.trapezoid(psd[signal_mask], freqs[signal_mask])) if signal_mask.any() else 0.0
        noise_power = float(np.trapezoid(psd[noise_mask], freqs[noise_mask])) if noise_mask.any() else 1.0
        if noise_power > 0:
            snrs.append(signal_power / noise_power)
    return float(np.mean(snrs)) if snrs else 0.0


def compute_peak_power(
    haemo_data: np.ndarray,
    sfreq: float,
) -> float:
    """Compute the peak spectral density across all hemoglobin channels.

    Parameters
    ----------
    haemo_data : np.ndarray
        Hemoglobin time-series, shape ``(channels, time)`` or ``(time,)``.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Maximum PSD value across all channels and frequencies.
    """
    if haemo_data.ndim == 1:
        haemo_data = haemo_data[None, :]

    peak = 0.0
    for ch_data in haemo_data:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(len(ch_data), int(8 * sfreq)))
        ch_peak = float(np.max(psd[freqs > 0]))  # exclude DC
        peak = max(peak, ch_peak)
    return peak


# ---------------------------------------------------------------------------
# Quality tier classification
# ---------------------------------------------------------------------------


def classify_quality_tier(
    sci_per_pair: Dict[str, float],
    scan_snr: Optional[float],
    cardiac_per_pair: Optional[Dict[str, bool]] = None,
    scan_passed: bool = True,
    thresholds: Optional[QualityTierThresholds] = None,
) -> str:
    """Classify a recording into a quality tier.

    Tiers are evaluated top-down: a recording gets the *highest* tier whose
    thresholds it meets.

    Parameters
    ----------
    sci_per_pair : dict
        ``{pair_name: sci_value}`` from :func:`compute_scalp_coupling_index`.
    scan_snr : float or None
        Scan-level SNR from :func:`compute_scan_snr`.  If None (scan-level
        checks were skipped), tier is capped at "standard".
    cardiac_per_pair : dict or None
        ``{pair_name: True/False}`` from :func:`_cardiac_by_pair`.
    scan_passed : bool
        Whether the recording passed the base QC check.
    thresholds : QualityTierThresholds, optional
        Custom thresholds; defaults to class defaults.

    Returns
    -------
    str
        One of :data:`QUALITY_TIERS`: ``"gold"``, ``"standard"``,
        ``"salvageable"``, or ``"rejected"``.
    """
    if thresholds is None:
        thresholds = QualityTierThresholds()

    if not sci_per_pair:
        return "rejected"

    mean_sci = float(np.mean(list(sci_per_pair.values())))
    all_cardiac = (
        all(cardiac_per_pair.values())
        if cardiac_per_pair
        else False
    )
    snr = scan_snr if scan_snr is not None else 0.0

    # Gold: pristine low-motion data
    if (
        mean_sci >= thresholds.gold_sci
        and snr >= thresholds.gold_snr
        and (all_cardiac or not thresholds.gold_require_all_cardiac)
    ):
        return "gold"

    # Standard: normal QC pass
    if mean_sci >= thresholds.standard_sci and snr >= thresholds.standard_snr:
        return "standard"

    # Salvageable: high-motion but not hopeless
    if mean_sci >= thresholds.salvageable_sci and snr >= thresholds.salvageable_snr:
        return "salvageable"

    return "rejected"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_quality_control(
    raw_scan,
    preprocessed_scan=None,
    *,
    sci_threshold: float = 0.5,
    snr_threshold: float = 5.0,
    cardiac_band: Tuple[float, float] = (0.8, 1.5),
    cardiac_peak_ratio: float = 2.0,
    require_cardiac: bool = True,
    hrf_band: Tuple[float, float] = (0.01, 0.2),
    peak_power_low: Optional[float] = None,
    peak_power_high: Optional[float] = None,
) -> QualityReport:
    """Run the full QC pipeline on a single recording.

    Parameters
    ----------
    raw_scan : mne.io.Raw
        Raw fNIRS recording (optical intensity), used for SCI and cardiac checks.
    preprocessed_scan : mne.io.Raw, optional
        Preprocessed hemoglobin recording, used for SNR and peak-power checks.
        If None, those checks are skipped.
    sci_threshold : float
        Minimum SCI to keep a channel (0–1). Default 0.5 (lenient).
        The literature recommends 0.75–0.95 for strict QC.
    snr_threshold : float
        Minimum scan-level SNR (linear). Scans below this are rejected.
    cardiac_band : tuple of float
        Frequency range for the cardiac signal.
    cardiac_peak_ratio : float
        Minimum ratio of cardiac peak to median PSD.
    require_cardiac : bool
        If True, channels without a cardiac signal are rejected.
    hrf_band : tuple of float
        Frequency range for the hemodynamic response (signal band for SNR).
    peak_power_low : float, optional
        Minimum acceptable peak PSD. Scans below this are noise-dominated.
    peak_power_high : float, optional
        Maximum acceptable peak PSD. Scans above this have motion artifacts.

    Returns
    -------
    QualityReport
    """
    reasons: List[str] = []

    # -- Channel-level: SCI --------------------------------------------------
    sci_per_pair = compute_scalp_coupling_index(raw_scan, cardiac_band=cardiac_band)
    channel_mask = {pair: sci >= sci_threshold for pair, sci in sci_per_pair.items()}
    n_sci_rejected = sum(1 for v in channel_mask.values() if not v)
    if n_sci_rejected:
        reasons.append(
            f"SCI < {sci_threshold}: {n_sci_rejected}/{len(channel_mask)} pairs rejected"
        )
    logger.info(
        "SCI check (threshold=%.2f): %d/%d pairs passed",
        sci_threshold,
        sum(channel_mask.values()),
        len(channel_mask),
    )

    # -- Channel-level: cardiac presence --------------------------------------
    cardiac_per_ch = check_cardiac_presence(
        raw_scan, cardiac_band=cardiac_band, min_peak_ratio=cardiac_peak_ratio
    )
    cardiac_per_pair = _cardiac_by_pair(cardiac_per_ch)
    if require_cardiac:
        for pair, has_cardiac in cardiac_per_pair.items():
            if pair in channel_mask and not has_cardiac:
                channel_mask[pair] = False
        n_cardiac_rejected = sum(
            1 for p, v in cardiac_per_pair.items() if p in channel_mask and not v
        )
        if n_cardiac_rejected:
            reasons.append(
                f"No cardiac signal: {n_cardiac_rejected} additional pairs rejected"
            )
    logger.info(
        "Cardiac check: %d/%d pairs have detectable heartbeat",
        sum(cardiac_per_pair.values()),
        len(cardiac_per_pair),
    )

    n_before = len(channel_mask)
    n_after = sum(channel_mask.values())

    # Scan fails if all channels rejected
    scan_passed = n_after > 0
    if not scan_passed:
        reasons.append("All channels rejected by SCI / cardiac checks")

    # -- Scan-level: SNR & peak power (on preprocessed data) ------------------
    scan_snr: Optional[float] = None
    scan_peak: Optional[float] = None

    if preprocessed_scan is not None and scan_passed:
        haemo_data = preprocessed_scan.get_data()  # (channels, time)
        haemo_sfreq = float(preprocessed_scan.info["sfreq"])

        scan_snr = compute_scan_snr(haemo_data, haemo_sfreq, hrf_band=hrf_band)
        if scan_snr < snr_threshold:
            scan_passed = False
            reasons.append(f"Scan SNR {scan_snr:.2f} < threshold {snr_threshold}")
        logger.info("Scan SNR: %.2f (threshold=%.1f)", scan_snr, snr_threshold)

        scan_peak = compute_peak_power(haemo_data, haemo_sfreq)
        if peak_power_low is not None and scan_peak < peak_power_low:
            scan_passed = False
            reasons.append(
                f"Peak power {scan_peak:.4g} < low bound {peak_power_low:.4g} (noise-dominated)"
            )
        if peak_power_high is not None and scan_peak > peak_power_high:
            scan_passed = False
            reasons.append(
                f"Peak power {scan_peak:.4g} > high bound {peak_power_high:.4g} (motion artifact)"
            )
        if peak_power_low is not None or peak_power_high is not None:
            logger.info(
                "Peak power: %.4g (bounds: [%s, %s])",
                scan_peak,
                str(peak_power_low),
                str(peak_power_high),
            )

    # Classify quality tier based on aggregate metrics
    tier = classify_quality_tier(
        sci_per_pair=sci_per_pair,
        scan_snr=scan_snr,
        cardiac_per_pair=cardiac_per_pair,
        scan_passed=scan_passed,
    )
    logger.info("Quality tier: %s", tier)

    return QualityReport(
        scan_passed=scan_passed,
        n_channels_before=n_before,
        n_channels_after=n_after,
        channel_mask=channel_mask,
        scan_snr=scan_snr,
        peak_power=scan_peak,
        channel_sci=sci_per_pair,
        cardiac_present=cardiac_per_pair,
        rejection_reasons=reasons,
        quality_tier=tier,
    )


# ---------------------------------------------------------------------------
# QC Cache — persist QC results so each recording is only checked once
# ---------------------------------------------------------------------------

import csv
import os
import threading

_qc_cache_lock = threading.Lock()


def load_qc_cache(cache_path: str) -> Dict[str, Dict]:
    """Load QC cache from CSV. Returns {fnirs_path: {scan_passed, quality_tier, scan_snr, ...}}."""
    cache = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache[row["fnirs_path"]] = {
                "scan_passed": row["scan_passed"].lower() == "true",
                "quality_tier": row.get("quality_tier", "unknown"),
                "scan_snr": float(row["scan_snr"]) if row.get("scan_snr") else None,
                "n_channels_before": int(row.get("n_channels_before", 0)),
                "n_channels_after": int(row.get("n_channels_after", 0)),
                "rejection_reasons": row.get("rejection_reasons", ""),
            }
    return cache


def save_qc_result(
    cache_path: str,
    fnirs_path: str,
    report: "QualityReport",
) -> None:
    """Append a single QC result to the cache CSV (thread-safe)."""
    with _qc_cache_lock:
        write_header = not os.path.exists(cache_path)
        with open(cache_path, "a", newline="") as f:
            fieldnames = [
                "fnirs_path", "scan_passed", "quality_tier", "scan_snr",
                "n_channels_before", "n_channels_after", "rejection_reasons",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "fnirs_path": fnirs_path,
                "scan_passed": str(report.scan_passed),
                "quality_tier": report.quality_tier,
                "scan_snr": f"{report.scan_snr:.4f}" if report.scan_snr is not None else "",
                "n_channels_before": report.n_channels_before,
                "n_channels_after": report.n_channels_after,
                "rejection_reasons": "; ".join(report.rejection_reasons),
            })

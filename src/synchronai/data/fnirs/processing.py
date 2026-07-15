"""
fNIRS I/O + preprocessing helpers.

This module intentionally stays lightweight: it loads fNIRS recordings with MNE,
then runs HRfunc's `preprocess_fnirs` (required by project conventions).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hrfunc
import mne
import numpy as np

from synchronai.utils.logging import get_logger
from synchronai.utils.trace import trace

@dataclass(frozen=True)
class HemoglobinMeta:
    sfreq_hz: float
    pair_names: List[str]
    hb_types: List[str]
    raw_channel_names: List[str]

def read_raw_fnirs(fnirs_path: str):
    logger = get_logger(__name__)
    if os.environ.get("SYNCHRONAI_VERBOSEE") == "1":
        mne.set_log_level("DEBUG")

    logger.debug("Reading fNIRS input: %s", fnirs_path)
    if os.path.isdir(fnirs_path):
        raw = mne.io.read_raw_nirx(fnirs_path, preload=True, verbose="ERROR")
        logger.debug(
            "Loaded NIRx raw: channels=%d, n_times=%d, sfreq=%.2f",
            len(raw.ch_names),
            raw.n_times,
            float(raw.info.get("sfreq", 0.0)),
        )
        return raw

    lower = fnirs_path.lower()
    if lower.endswith(".fif"):
        raw = mne.io.read_raw_fif(fnirs_path, preload=True, verbose="ERROR")
        logger.debug(
            "Loaded FIF raw: channels=%d, n_times=%d, sfreq=%.2f",
            len(raw.ch_names),
            raw.n_times,
            float(raw.info.get("sfreq", 0.0)),
        )
        return raw
    if lower.endswith(".snirf"):
        raw = mne.io.read_raw_snirf(fnirs_path, preload=True, verbose="ERROR")
        logger.debug(
            "Loaded SNIRF raw: channels=%d, n_times=%d, sfreq=%.2f",
            len(raw.ch_names),
            raw.n_times,
            float(raw.info.get("sfreq", 0.0)),
        )
        return raw

    raise ValueError(f"Unsupported fNIRS path: {fnirs_path}")

def load_fnirs(fnirs_path: str, *, deconvolution: bool = False):
    """
    Load + preprocess a single fNIRS recording.

    Returns the preprocessed MNE Raw instance (typically hemoglobin channels).
    """
    logger = get_logger(__name__)
    trace(f"load_fnirs: reading raw {fnirs_path}")
    raw_scan = read_raw_fnirs(fnirs_path)
    trace("load_fnirs: raw read complete")

    # Validate raw data before preprocessing to catch issues early
    if raw_scan.n_times == 0:
        raise ValueError(f"Empty recording (n_times=0): {fnirs_path}")
    if len(raw_scan.ch_names) == 0:
        raise ValueError(f"No channels in recording: {fnirs_path}")

    trace(f"load_fnirs: validated raw scan (n_times={raw_scan.n_times}, channels={len(raw_scan.ch_names)})")

    logger.debug("Running HRfunc preprocess_fnirs (deconvolution=%s)", str(deconvolution))
    trace(f"load_fnirs: running hrfunc.preprocess_fnirs deconvolution={deconvolution}")

    # Wrap HRfunc preprocessing with error handling to catch C-extension segfaults
    try:
        processed = hrfunc.preprocess_fnirs(raw_scan, deconvolution=deconvolution)
    except Exception as e:
        logger.error(f"HRfunc preprocessing failed for {fnirs_path}: {type(e).__name__}: {e}")
        trace(f"load_fnirs: hrfunc.preprocess_fnirs FAILED - {type(e).__name__}: {e}")
        raise RuntimeError(
            f"HRfunc preprocessing failed for {fnirs_path}. "
            f"This may indicate corrupted data or C-extension issues. Error: {e}"
        ) from e

    trace("load_fnirs: hrfunc.preprocess_fnirs complete")

    # Check if preprocessing returned None (all channels bad)
    if processed is None:
        logger.warning(f"Preprocessing returned None (all channels bad): {fnirs_path}")
        raise ValueError(f"All channels bad in recording: {fnirs_path}")

    # Validate preprocessed data
    if processed.n_times == 0:
        raise ValueError(f"Preprocessing resulted in empty data (n_times=0): {fnirs_path}")
    if len(processed.ch_names) == 0:
        raise ValueError(f"Preprocessing resulted in no channels: {fnirs_path}")

    try:
        logger.debug(
            "Preprocessed scan: channels=%d, n_times=%d, sfreq=%.2f",
            len(processed.ch_names),
            processed.n_times,
            float(processed.info.get("sfreq", 0.0)),
        )
    except Exception:
        logger.debug("Preprocessed scan ready.")
    return processed


def process_fnirs(scan, deconvolution: bool = False):
    """Run HRfunc montage + preprocessing on an MNE Raw instance."""
    hrfunc.montage(scan)
    return hrfunc.preprocess_fnirs(scan, deconvolution=deconvolution)


def extract_hemoglobin_pairs(preprocessed_scan) -> Tuple[np.ndarray, HemoglobinMeta]:
    """
    Convert a preprocessed MNE Raw to a dense hemoglobin tensor.

    Returns:
      x: float32 array shaped (time, pairs, hb_types)
      meta: metadata describing axes.
    """
    logger = get_logger(__name__)
    data = preprocessed_scan.get_data().astype(np.float32)  # (channels, time)
    ch_names = list(preprocessed_scan.ch_names)
    sfreq = float(preprocessed_scan.info.get("sfreq", 1.0))
    logger.debug("Extracting hemoglobin pairs: channels=%d, sfreq=%.2f", len(ch_names), sfreq)

    try:
        ch_types = list(preprocessed_scan.get_channel_types())
    except Exception:
        ch_types = ["misc"] * len(ch_names)

    hb_candidates = {"hbo", "hbr"}
    hb_indices: Dict[str, List[int]] = {hb: [] for hb in hb_candidates}

    for idx, (name, ch_type) in enumerate(zip(ch_names, ch_types)):
        ch_type_lower = str(ch_type).lower()
        name_lower = name.lower()
        if ch_type_lower in hb_candidates:
            hb_indices[ch_type_lower].append(idx)
        elif name_lower.endswith((" hbo", "_hbo", "-hbo")):
            hb_indices["hbo"].append(idx)
        elif name_lower.endswith((" hbr", "_hbr", "-hbr")):
            hb_indices["hbr"].append(idx)

    # If HRfunc didn't yield explicit HbO/HbR types, fall back to "all channels".
    if not hb_indices["hbo"] and not hb_indices["hbr"]:
        x = data.T[:, :, None]  # (time, channels, 1)
        meta = HemoglobinMeta(
            sfreq_hz=sfreq,
            pair_names=ch_names,
            hb_types=["unknown"],
            raw_channel_names=ch_names,
        )
        return x, meta

    def _pair_key(ch_name: str) -> str:
        lowered = ch_name.lower()
        for suffix in (" hbo", " hbr", "_hbo", "_hbr", "-hbo", "-hbr"):
            if lowered.endswith(suffix):
                return ch_name[: -len(suffix)]
        return ch_name

    # Build paired tensors (only keep pairs where we have both HbO and HbR).
    hbo_by_key = {_pair_key(ch_names[i]): i for i in hb_indices["hbo"]}
    hbr_by_key = {_pair_key(ch_names[i]): i for i in hb_indices["hbr"]}
    shared_keys = sorted(set(hbo_by_key) & set(hbr_by_key))
    if not shared_keys:
        raise ValueError(
            "Could not form HbO/HbR pairs from preprocessed channels. "
            "Check HRfunc preprocessing output."
        )

    pair_names = shared_keys

    # Validate array shapes before stacking to prevent segfaults from mismatched dimensions
    hbo_arrays = [data[hbo_by_key[k]] for k in shared_keys]
    hbr_arrays = [data[hbr_by_key[k]] for k in shared_keys]

    # Check all arrays have the same shape
    if hbo_arrays:
        first_shape = hbo_arrays[0].shape
        for i, arr in enumerate(hbo_arrays):
            if arr.shape != first_shape:
                raise ValueError(
                    f"HbO channel shape mismatch: channel {i} has shape {arr.shape}, "
                    f"expected {first_shape}. This may indicate corrupted data."
                )
        for i, arr in enumerate(hbr_arrays):
            if arr.shape != first_shape:
                raise ValueError(
                    f"HbR channel shape mismatch: channel {i} has shape {arr.shape}, "
                    f"expected {first_shape}. This may indicate corrupted data."
                )

    hbo_stack = np.stack(hbo_arrays, axis=0)  # (pairs, time)
    hbr_stack = np.stack(hbr_arrays, axis=0)  # (pairs, time)

    # Validate stacked shapes before final transformation
    if hbo_stack.shape != hbr_stack.shape:
        raise ValueError(
            f"HbO and HbR stack shape mismatch: HbO={hbo_stack.shape}, HbR={hbr_stack.shape}. "
            f"This may indicate corrupted data or preprocessing issues."
        )

    x = np.stack([hbo_stack, hbr_stack], axis=-1).transpose(1, 0, 2)  # (time, pairs, 2)

    meta = HemoglobinMeta(
        sfreq_hz=sfreq,
        pair_names=pair_names,
        hb_types=["hbo", "hbr"],
        raw_channel_names=ch_names,
    )
    return x.astype(np.float32), meta

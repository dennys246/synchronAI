"""Shared windowing logic for fNIRS recordings.

Single source of truth for how a ``(time, ...)`` recording is sliced into
fixed-length windows. Used by both generative pretraining
(:func:`synchronai.data.fnirs.dataset.load_training_windows`) and encoder
feature extraction (``scripts/extract_fnirs_features.window_recording``) so the
two windowing schemes cannot silently drift apart.
"""

from __future__ import annotations

from typing import List


def tile_window_starts(
    time_len: int,
    window_len: int,
    stride_samples: int,
    phase_offset: int = 0,
) -> List[int]:
    """Return start indices for tiling a time signal into fixed-length windows.

    Non-overlapping tiling uses ``stride_samples == window_len``. A positive
    ``phase_offset`` shifts the tiling grid forward — used as a per-epoch
    augmentation during generative training so events land at varied positions
    within a window — and is clamped so at least one full window still fits.

    Returns an empty list when no full window fits (``time_len < window_len``);
    the caller is responsible for zero-padding a single short window in that
    case (the behaviour both call sites already implement).
    """
    if window_len <= 0 or stride_samples <= 0:
        raise ValueError(
            "window_len and stride_samples must be positive "
            f"(got window_len={window_len}, stride_samples={stride_samples})"
        )
    if time_len < window_len:
        return []
    # Clamp the offset so a short-but-fitting recording still yields one window.
    start = min(max(0, int(phase_offset)), time_len - window_len)
    starts: List[int] = []
    while start + window_len <= time_len:
        starts.append(start)
        start += stride_samples
    return starts

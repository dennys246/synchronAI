"""Wavelet transform coherence (Morlet), Grinsted-style smoothing.

Replaces the Phase-1 feasibility implementation
(scripts/fnirs_synchrony_phase1_feasibility.py), which had two defects that
made its single-dyad null result uninterpretable:

1. Fixed 10 s time-smoothing against periods up to 100 s saturated coherence
   (surrogate-mean WTC ~0.66). Coherence is only meaningful when the smoothing
   window scales with the analyzed period — here we use the Grinsted et al.
   2004 convention: Gaussian time smoothing with sigma = scale, and a boxcar
   over 0.6 octave in the scale direction.
2. A global edge-drop at the lowest frequency's cone of influence discarded
   valid samples at high frequencies. Here the COI is applied per scale.

scipy-only by design: ml-env has no wavelet library and does not need one.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.fft import fft, ifft

from synchronai.utils.logging import get_logger

# Morlet omega0=6: standard choice, Fourier period ~= 1.03 * scale.
DEFAULT_OMEGA0 = 6.0


def _scales_for_freqs(freqs: np.ndarray, omega0: float) -> np.ndarray:
    return (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * freqs)


def morlet_cwt(
    x: np.ndarray, freqs: np.ndarray, fs: float, omega0: float = DEFAULT_OMEGA0
) -> np.ndarray:
    """FFT-based Morlet CWT. Returns complex (n_freqs, n_samples)."""
    n = x.shape[-1]
    X = fft(x, n=n)
    omega = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / fs)
    scales = _scales_for_freqs(freqs, omega0)
    out = np.empty((len(freqs), n), dtype=np.complex128)
    for i, s in enumerate(scales):
        norm = np.sqrt(2 * np.pi * s * fs) * (np.pi**-0.25)
        h = (omega * s > 0).astype(float)
        psi_f = norm * h * np.exp(-((s * omega - omega0) ** 2) / 2)
        out[i] = ifft(X * psi_f, n=n)
    return out


def _gaussian_smooth_time(W: np.ndarray, scales: np.ndarray, fs: float) -> np.ndarray:
    """Per-scale Gaussian time smoothing, sigma = scale (Grinsted 2004).

    FFT-based: multiply each row's spectrum by the Gaussian's transfer
    function exp(-0.5 * (sigma * omega)^2).
    """
    n = W.shape[-1]
    omega = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / fs)
    out = np.empty(W.shape, dtype=np.complex128)
    for i, s in enumerate(scales):
        kernel_f = np.exp(-0.5 * (s * omega) ** 2)
        out[i] = ifft(fft(W[i], n=n) * kernel_f, n=n)
    return out if np.iscomplexobj(W) else out.real


def _boxcar_smooth_scale(W: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Boxcar smoothing over 0.6 octave in the scale direction.

    Requires geometrically spaced freqs (constant bins per octave).
    """
    n_freqs = len(freqs)
    if n_freqs < 2:
        return W
    octaves = np.log2(freqs[-1] / freqs[0])
    per_octave = (n_freqs - 1) / octaves
    win = int(round(0.6 * per_octave))
    win = max(1, win) | 1  # odd, >= 1
    if win == 1:
        return W
    kernel = np.ones(win) / win
    pad = win // 2
    padded = np.concatenate(
        [W[:1].repeat(pad, axis=0), W, W[-1:].repeat(pad, axis=0)], axis=0
    )
    out = np.empty_like(W)
    for t in range(0, W.shape[-1], 100_000):  # chunk to bound memory
        sl = slice(t, min(t + 100_000, W.shape[-1]))
        seg = padded[:, sl]
        out[:, sl] = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="valid"), 0, seg
        )
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    if sd < 1e-30:
        return x - mu
    return (x - mu) / sd


def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    omega0: float = DEFAULT_OMEGA0,
) -> np.ndarray:
    """Magnitude-squared wavelet coherence (Grinsted 2004 Eq. 2-3).

        R^2(s,t) = |S(s^-1 W^XY)|^2 / (S(s^-1 |W^X|^2) * S(s^-1 |W^Y|^2))

    S = Gaussian(sigma=scale) in time then 0.6-octave boxcar in scale.
    freqs must be geometrically spaced. Returns (n_freqs, n_samples).
    """
    x = _zscore(np.asarray(x, dtype=np.float64))
    y = _zscore(np.asarray(y, dtype=np.float64))
    scales = _scales_for_freqs(freqs, omega0)

    Wx = morlet_cwt(x, freqs, fs, omega0)
    Wy = morlet_cwt(y, freqs, fs, omega0)
    inv_s = (1.0 / scales)[:, None]

    Wxy = _gaussian_smooth_time(Wx * np.conj(Wy) * inv_s, scales, fs)
    Wxx = _gaussian_smooth_time((np.abs(Wx) ** 2) * inv_s, scales, fs).real
    Wyy = _gaussian_smooth_time((np.abs(Wy) ** 2) * inv_s, scales, fs).real

    Wxy = _boxcar_smooth_scale(Wxy, freqs)
    Wxx = _boxcar_smooth_scale(Wxx, freqs)
    Wyy = _boxcar_smooth_scale(Wyy, freqs)

    denom = Wxx * Wyy
    wtc = (np.abs(Wxy) ** 2) / np.maximum(denom, np.finfo(denom.dtype).tiny)
    return np.clip(wtc.real, 0.0, 1.0)


def coi_mask(
    n_samples: int, fs: float, freqs: np.ndarray, omega0: float = DEFAULT_OMEGA0
) -> np.ndarray:
    """Boolean (n_freqs, n_samples) mask, True where OUTSIDE the cone of
    influence (i.e. valid). Per-scale edge width = sqrt(2) * scale."""
    scales = _scales_for_freqs(freqs, omega0)
    edge = np.ceil(np.sqrt(2) * scales * fs).astype(int)  # samples, per scale
    idx = np.arange(n_samples)
    return (idx[None, :] >= edge[:, None]) & (idx[None, :] < n_samples - edge[:, None])


def band_mean_wtc(
    wtc: np.ndarray,
    freqs: np.ndarray,
    band_hz: tuple[float, float],
    valid: np.ndarray,
) -> tuple[float, np.ndarray]:
    """COI-aware band summary.

    Returns (scalar mean over the band's valid region, per-sample band-mean
    time series with NaN where no in-band scale is valid).
    """
    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not in_band.any():
        raise ValueError(f"No frequencies inside band {band_hz}")
    w = wtc[in_band]
    v = valid[in_band]
    masked = np.where(v, w, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        series = np.nanmean(masked, axis=0)  # all-NaN columns (inside COI) -> NaN
        scalar = float(np.nanmean(masked))
    return scalar, series


def drop_duplicate_channels(
    signals: np.ndarray, pair_names: list[str], threshold: float = 0.9999
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop channels that are near-exact copies of an earlier channel.

    Phase-1 finding: some recordings contain S-D pairs identical to ~8
    decimals (probably upstream bad-channel interpolation copying a
    neighbor). signals is (T, P); returns (signals_kept, names_kept,
    names_dropped).
    """
    logger = get_logger(__name__)
    P = signals.shape[1]
    if P < 2:
        return signals, list(pair_names), []
    finite = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    std = finite.std(axis=0)
    std[std < 1e-30] = 1.0
    z = (finite - finite.mean(axis=0)) / std
    corr = (z.T @ z) / z.shape[0]
    keep, dropped = [], []
    for j in range(P):
        if any(abs(corr[j, k]) > threshold for k in keep):
            dropped.append(pair_names[j])
        else:
            keep.append(j)
    if dropped:
        logger.warning(
            "Dropped %d near-duplicate channel(s) (|r|>%.4f): %s",
            len(dropped), threshold, dropped,
        )
    return signals[:, keep], [pair_names[j] for j in keep], dropped

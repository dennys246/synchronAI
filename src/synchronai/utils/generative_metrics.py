"""
Generative quality metrics for fNIRS diffusion models.

Implements FID (Fréchet Inception Distance adapted for time-series) and
MMD (Maximum Mean Discrepancy) to measure distributional similarity between
real and generated fNIRS windows.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def _flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Flatten (n_samples, time, features) -> (n_samples, time*features)."""
    if windows.ndim == 3:
        return windows.reshape(windows.shape[0], -1)
    return windows


def compute_fid(real: np.ndarray, generated: np.ndarray) -> float:
    """
    Compute the Fréchet distance between real and generated window distributions.

    Operates directly on flattened time-series windows (no learned feature
    extractor). Measures both mean shift and covariance divergence.

    Args:
        real: Array of shape (n_samples, time, features)
        generated: Array of shape (n_samples, time, features)

    Returns:
        Fréchet distance (lower is better, 0 = identical distributions).
    """
    r = _flatten_windows(real).astype(np.float64)
    g = _flatten_windows(generated).astype(np.float64)

    mu_r, mu_g = r.mean(axis=0), g.mean(axis=0)
    sigma_r = np.cov(r, rowvar=False)
    sigma_g = np.cov(g, rowvar=False)

    diff = mu_r - mu_g
    mean_term = diff @ diff

    # Matrix square root of product of covariances
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    # Numerical stability: discard tiny imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_term + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return max(float(fid), 0.0)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix between x and y."""
    # x: (n, d), y: (m, d) -> (n, m)
    xx = np.sum(x ** 2, axis=1, keepdims=True)  # (n, 1)
    yy = np.sum(y ** 2, axis=1, keepdims=True)  # (m, 1)
    dists_sq = xx + yy.T - 2.0 * (x @ y.T)     # (n, m)
    return np.exp(-dists_sq / (2.0 * bandwidth ** 2))


def compute_mmd(real: np.ndarray, generated: np.ndarray, bandwidths: list[float] | None = None) -> float:
    """
    Compute MMD² (Maximum Mean Discrepancy) between real and generated windows.

    Uses a mixture of RBF kernels at multiple bandwidths for robustness
    across different scales.

    Args:
        real: Array of shape (n_samples, time, features)
        generated: Array of shape (n_samples, time, features)
        bandwidths: RBF kernel bandwidths. If None, uses median heuristic
                    with multipliers [0.5, 1.0, 2.0].

    Returns:
        MMD² estimate (lower is better, 0 = identical distributions).
    """
    r = _flatten_windows(real).astype(np.float64)
    g = _flatten_windows(generated).astype(np.float64)

    if bandwidths is None:
        # Median heuristic: use median pairwise distance as base bandwidth
        subset = np.vstack([r[:min(200, len(r))], g[:min(200, len(g))]])
        dists = np.sum((subset[:, None, :] - subset[None, :, :]) ** 2, axis=-1)
        median_dist = np.sqrt(max(np.median(dists[dists > 0]), 1e-8))
        bandwidths = [median_dist * m for m in (0.5, 1.0, 2.0)]

    mmd_sq = 0.0
    for bw in bandwidths:
        k_rr = _rbf_kernel(r, r, bw)
        k_gg = _rbf_kernel(g, g, bw)
        k_rg = _rbf_kernel(r, g, bw)
        mmd_sq += k_rr.mean() + k_gg.mean() - 2.0 * k_rg.mean()

    # Average over bandwidths
    mmd_sq /= len(bandwidths)
    return max(float(mmd_sq), 0.0)

"""
Visualization utilities for fNIRS hemoglobin signals.

Provides plotting functions for visualizing generated and real hemoglobin time series.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_hemoglobin_signal(
    data: np.ndarray,
    *,
    sfreq_hz: float,
    pair_names: Optional[List[str]] = None,
    hb_types: Optional[List[str]] = None,
    title: str = "fNIRS Hemoglobin Signal",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 8),
    max_pairs: int = 4,
) -> None:
    """
    Plot hemoglobin time series in the style typically used for fNIRS signals.

    Args:
        data: Array of shape (time, pairs, hb) or (time, features)
        sfreq_hz: Sampling frequency in Hz
        pair_names: Optional list of channel pair names
        hb_types: Optional list of hemoglobin types (e.g., ["hbo", "hbr"])
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        max_pairs: Maximum number of pairs to plot (to avoid cluttered plots)
    """
    # Handle both 3D (time, pairs, hb) and 2D (time, features) data
    if data.ndim == 2:
        # Reshape (time, features) -> (time, pairs, hb)
        n_time, n_features = data.shape
        if hb_types is None:
            hb_types = ["hbo", "hbr"]
        n_hb = len(hb_types)
        if n_features % n_hb != 0:
            raise ValueError(
                f"Cannot reshape features={n_features} with hb_types={len(hb_types)}"
            )
        n_pairs = n_features // n_hb
        data = data.reshape(n_time, n_pairs, n_hb)
    else:
        n_time, n_pairs, n_hb = data.shape

    # Use default names if not provided
    if pair_names is None:
        pair_names = [f"Pair_{i+1}" for i in range(n_pairs)]
    if hb_types is None:
        hb_types = ["hbo", "hbr"] if n_hb == 2 else [f"hb{i}" for i in range(n_hb)]

    # Limit number of pairs to plot
    n_pairs_to_plot = min(n_pairs, max_pairs)

    # Create time axis in seconds
    time_seconds = np.arange(n_time) / sfreq_hz

    # Set up colors: HbO is typically red, HbR is typically blue
    colors = {
        "hbo": "#d62728",  # red
        "hbr": "#1f77b4",  # blue
    }
    # Fallback for other hemoglobin types
    default_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(n_pairs_to_plot, 1, figsize=figsize, sharex=True)
    if n_pairs_to_plot == 1:
        axes = [axes]

    for pair_idx in range(n_pairs_to_plot):
        ax = axes[pair_idx]
        pair_name = pair_names[pair_idx] if pair_idx < len(pair_names) else f"Pair_{pair_idx+1}"

        for hb_idx, hb_type in enumerate(hb_types):
            signal = data[:, pair_idx, hb_idx]

            # Get color for this hemoglobin type
            hb_lower = hb_type.lower()
            if hb_lower in colors:
                color = colors[hb_lower]
            else:
                color = default_colors[hb_idx % len(default_colors)]

            ax.plot(
                time_seconds,
                signal,
                label=hb_type.upper(),
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )

        ax.set_ylabel(f"{pair_name}\nConcentration (a.u.)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Set x-label only on bottom plot
    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_multiple_samples(
    samples: np.ndarray,
    *,
    sfreq_hz: float,
    pair_names: Optional[List[str]] = None,
    hb_types: Optional[List[str]] = None,
    title: str = "Generated fNIRS Samples",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 10),
    max_samples: int = 3,
    max_pairs: int = 2,
) -> None:
    """
    Plot multiple generated samples in a grid layout.

    Args:
        samples: Array of shape (n_samples, time, pairs, hb) or (n_samples, time, features)
        sfreq_hz: Sampling frequency in Hz
        pair_names: Optional list of channel pair names
        hb_types: Optional list of hemoglobin types
        title: Overall plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        max_samples: Maximum number of samples to plot
        max_pairs: Maximum number of pairs to plot per sample
    """
    n_samples = min(samples.shape[0], max_samples)

    # Handle both 4D and 3D data
    if samples.ndim == 3:
        # Reshape (n_samples, time, features) -> (n_samples, time, pairs, hb)
        n_samp, n_time, n_features = samples.shape
        if hb_types is None:
            hb_types = ["hbo", "hbr"]
        n_hb = len(hb_types)
        if n_features % n_hb != 0:
            raise ValueError(
                f"Cannot reshape features={n_features} with hb_types={len(hb_types)}"
            )
        n_pairs = n_features // n_hb
        samples = samples.reshape(n_samp, n_time, n_pairs, n_hb)
    else:
        _, n_time, n_pairs, n_hb = samples.shape

    # Use default names if not provided
    if pair_names is None:
        pair_names = [f"Pair_{i+1}" for i in range(n_pairs)]
    if hb_types is None:
        hb_types = ["hbo", "hbr"] if n_hb == 2 else [f"hb{i}" for i in range(n_hb)]

    n_pairs_to_plot = min(n_pairs, max_pairs)
    time_seconds = np.arange(n_time) / sfreq_hz

    # Colors
    colors = {
        "hbo": "#d62728",
        "hbr": "#1f77b4",
    }
    default_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(
        n_samples, n_pairs_to_plot, figsize=figsize, sharex=True, sharey=False
    )

    # Handle single sample or single pair cases
    if n_samples == 1 and n_pairs_to_plot == 1:
        axes = [[axes]]
    elif n_samples == 1:
        axes = [axes]
    elif n_pairs_to_plot == 1:
        axes = [[ax] for ax in axes]

    for sample_idx in range(n_samples):
        for pair_idx in range(n_pairs_to_plot):
            ax = axes[sample_idx][pair_idx]
            pair_name = pair_names[pair_idx] if pair_idx < len(pair_names) else f"Pair_{pair_idx+1}"

            for hb_idx, hb_type in enumerate(hb_types):
                signal = samples[sample_idx, :, pair_idx, hb_idx]

                hb_lower = hb_type.lower()
                if hb_lower in colors:
                    color = colors[hb_lower]
                else:
                    color = default_colors[hb_idx % len(default_colors)]

                ax.plot(
                    time_seconds,
                    signal,
                    label=hb_type.upper(),
                    color=color,
                    linewidth=1.0,
                    alpha=0.8,
                )

            # Labels
            if sample_idx == 0:
                ax.set_title(f"{pair_name}", fontsize=10)
            if pair_idx == 0:
                ax.set_ylabel(f"Sample {sample_idx+1}\nConc. (a.u.)", fontsize=9)
            if sample_idx == 0 and pair_idx == n_pairs_to_plot - 1:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # X-label on bottom row
    for pair_idx in range(n_pairs_to_plot):
        axes[-1][pair_idx].set_xlabel("Time (s)", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

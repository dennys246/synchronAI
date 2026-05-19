#!/usr/bin/env python3
"""Figure 3 — Representation redundancy bar chart.

Shows R²(video_repr → audio_repr) and R²(audio_repr → video_repr) across
three audio conditions, with error bars across the 5 CV folds. Demonstrates
that the model's audio pathway becomes a linearly-predictable function of
the video pathway regardless of input feature properties (transformer vs.
25-dim acoustic) or training regime (vanilla vs. modality dropout).

R² values are hardcoded from the prior runs of
scripts/diagnose_modality_repr_correlation.py. To regenerate, re-run that
script on the cv5 dirs and update the table below.

Usage:
    python scripts/paper_figures/plot_redundancy.py \\
        --output figures/multimodal_paper/fig3_redundancy.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# (condition_label, R²(V→A) mean, R²(V→A) std, R²(A→V) mean, R²(A→V) std)
# Std is across the 5 CV folds (computed by diagnose_modality_repr_correlation.py).
CONDITIONS = [
    ("WavLM-base-plus\n(vanilla training)",        0.6457, 0.1156, 0.9211, 0.0254),
    ("WavLM-base-plus\n(+ modality dropout 0.3)",  0.9798, 0.0076, 0.9463, 0.0251),
    ("eGeMAPS LLDs (25-dim)\n(vanilla training)",  0.9332, 0.0271, 0.9238, 0.0295),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    n = len(CONDITIONS)
    x = np.arange(n)
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))

    va_means = [c[1] for c in CONDITIONS]
    va_stds = [c[2] for c in CONDITIONS]
    av_means = [c[3] for c in CONDITIONS]
    av_stds = [c[4] for c in CONDITIONS]

    bars_va = ax.bar(
        x - bar_width / 2, va_means, bar_width, yerr=va_stds,
        capsize=4, color="#5DA5DA", edgecolor="#1F4060",
        label="R²(video_repr → audio_repr)",
    )
    bars_av = ax.bar(
        x + bar_width / 2, av_means, bar_width, yerr=av_stds,
        capsize=4, color="#FAA43A", edgecolor="#8C5A1E",
        label="R²(audio_repr → video_repr)",
    )

    # Annotate values
    for bars, means in [(bars_va, va_means), (bars_av, av_means)]:
        for b, m in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2, m + 0.018,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Linear R² between modality representations\n(mean ± std across 5 folds)",
                  fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.85, color="gray", linestyle=":", lw=1.0, alpha=0.6,
               label="High-redundancy threshold (R² = 0.85)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title(
        "Audio pathway becomes video-redundant regardless of audio features or training",
        fontsize=11, pad=10,
    )

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()

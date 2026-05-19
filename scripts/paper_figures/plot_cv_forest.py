#!/usr/bin/env python3
"""Figure 1 — Forest plot of CV mean val_AUC across all tested conditions.

Visualizes the convergence of every fusion architecture, audio feature family,
and training regime to the same CV mean (~0.72 ± 0.05 val_AUC), with the
single-modality ablations included for context. The horizontal extent of the
95% CIs shows fold-composition variance dominates condition-to-condition
variance.

Reads per-fold val_AUC from each runs/multimodal_features/*_cv5/fold_*/
history.json. Writes a single PDF.

Usage:
    python scripts/paper_figures/plot_cv_forest.py \\
        --output figures/multimodal_paper/fig1_cv_forest.pdf
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# (display name, run-dir tag, group). Order = display order top→bottom.
# Group names mirror the paper subsections so the figure reads top-down.
CONDITIONS = [
    # Multimodal variants
    ("V2 multimodal (WavLM-base-plus)",        "v2_baseline_v6_cv5",     "multimodal"),
    ("V2 multimodal + modality dropout",       "v2_moddropout03_cv5",    "multimodal"),
    ("V2 multimodal (eGeMAPS LLDs)",           "v2_prosodic_cv5",        "multimodal"),
    ("V3 multimodal (cross-attn on summaries)", "v3_h24_cv5",            "multimodal"),
    ("V4 multimodal (token-level cross-attn)",  "v4_baseline_cv5",       "multimodal"),
    # Single-modality ablations
    ("Video-only (audio zeroed)",              "v2_video_only_cv5",      "single"),
    ("Audio-only (video zeroed, WavLM)",       "v2_audio_only_cv5",      "single"),
]


def load_cv_aucs(run_dir: Path) -> list[float]:
    """Return per-fold best val_AUC from each fold_*/history.json."""
    aucs = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        hist_path = fold_dir / "history.json"
        if not hist_path.exists():
            continue
        h = json.load(open(hist_path))
        aucs.append(max(h["val_aucs"]))
    return aucs


def summarize(aucs: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, SE, ci_half_width) over folds."""
    arr = np.array(aucs)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    ci_half = 1.96 * se
    return mean, std, se, ci_half


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", default="runs/multimodal_features",
                        help="Parent dir containing the *_cv5 run dirs")
    parser.add_argument("--output", required=True, help="Output PDF path")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Gather data for each condition
    rows = []
    for display_name, run_tag, group in CONDITIONS:
        rd = runs_root / run_tag
        aucs = load_cv_aucs(rd)
        if not aucs:
            print(f"WARNING: no fold data found for {run_tag} — skipping")
            continue
        mean, std, se, ci = summarize(aucs)
        rows.append((display_name, group, mean, ci, len(aucs), aucs))
        print(f"  {display_name:<42}  n={len(aucs)}  AUC = {mean:.4f} ± {ci:.4f} (95% CI)")

    if not rows:
        raise SystemExit("No data found; check --runs-root.")

    # --- Plot ---
    # One row per condition, vertical layout (forest plot convention).
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    y_positions = list(range(len(rows)))[::-1]  # top→bottom matches list order
    group_color = {"multimodal": "#2E5C8A", "single": "#A33A3A"}

    for y, (display_name, group, mean, ci, n, aucs) in zip(y_positions, rows):
        color = group_color[group]
        # Error bar (95% CI)
        ax.errorbar([mean], [y], xerr=[ci], fmt="o", color=color,
                    markersize=7, capsize=4, lw=1.4, ecolor=color, mfc=color, mec=color)
        # Per-fold dots (lighter, behind)
        for a in aucs:
            ax.plot(a, y, "o", color=color, alpha=0.18, markersize=4, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("5-fold CV mean val_AUC (95% CI; individual folds shown as faded dots)", fontsize=10)
    ax.set_xlim(0.55, 0.85)
    ax.axvline(x=0.5, color="gray", linestyle=":", lw=1.0, alpha=0.6, label="Chance (0.5)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group separator
    multimodal_count = sum(1 for r in rows if r[1] == "multimodal")
    if multimodal_count and multimodal_count < len(rows):
        ax.axhline(
            y=y_positions[multimodal_count] + 0.5,
            color="gray", linestyle="-", lw=0.7, alpha=0.4,
        )

    # Subtle legend explaining the two groupings
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=group_color["multimodal"], lw=0,
               markersize=7, label="Multimodal variants"),
        Line2D([0], [0], marker="o", color=group_color["single"], lw=0,
               markersize=7, label="Single-modality ablations"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, frameon=False)

    ax.set_title(
        "Multimodal synchrony classifier: CV mean val_AUC across conditions",
        fontsize=11, pad=10,
    )

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()

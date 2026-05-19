#!/usr/bin/env python3
"""Figure 2 — Per-fold val_AUC trajectory across multimodal architectures.

Shows that fold rank is constant across architectures: fold 0 best for all,
fold 4 worst for all. The within-fold spread across architectures (vertical
gap between lines at any x) is small compared to the across-fold range
(horizontal swing of any single line). Visualizes "fold variance dominates
architecture variance" directly.

Reads per-fold val_AUC from each runs/multimodal_features/*_cv5/fold_*/
history.json. Writes a single PDF.

Usage:
    python scripts/paper_figures/plot_fold_rank.py \\
        --output figures/multimodal_paper/fig2_fold_rank.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Architectures to compare (multimodal only — single-modality runs not relevant here).
# (display name, run-dir tag, line style)
ARCHS = [
    ("V2 (concat)",                       "v2_baseline_v6_cv5",    "-",  "#2E5C8A"),
    ("V3 (cross-attn on summaries, h=24)", "v3_h24_cv5",            "--", "#5DA5DA"),
    ("V4 (token-level cross-attn)",        "v4_baseline_cv5",       ":",  "#7BAFD4"),
    ("V2 + modality dropout",              "v2_moddropout03_cv5",   "-.", "#1F4060"),
    ("V2 (eGeMAPS audio)",                 "v2_prosodic_cv5",       "-",  "#A33A3A"),
]


def load_per_fold_aucs(run_dir: Path, n_folds: int = 5) -> list[float]:
    """Return per-fold best val_AUC, indexed by fold_idx 0..n_folds-1."""
    aucs = [None] * n_folds
    for fold_dir in sorted(run_dir.glob("fold_*")):
        # fold_dir.name = "fold_0", "fold_1", etc.
        try:
            idx = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx >= n_folds:
            continue
        hist_path = fold_dir / "history.json"
        if not hist_path.exists():
            continue
        h = json.load(open(hist_path))
        aucs[idx] = max(h["val_aucs"])
    return aucs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", default="runs/multimodal_features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    fold_indices = list(range(5))
    any_data = False

    for display_name, run_tag, linestyle, color in ARCHS:
        rd = runs_root / run_tag
        per_fold = load_per_fold_aucs(rd, n_folds=5)
        if all(v is None for v in per_fold):
            print(f"WARNING: no data for {run_tag} — skipping")
            continue
        # Drop missing folds for plotting but keep order
        xs = [i for i, v in enumerate(per_fold) if v is not None]
        ys = [v for v in per_fold if v is not None]
        ax.plot(xs, ys, linestyle=linestyle, marker="o", color=color,
                lw=1.6, markersize=6.5, label=display_name, alpha=0.92)
        any_data = True
        mean_auc = float(np.mean(ys))
        print(f"  {display_name:<42}  per-fold AUCs={[round(y, 4) for y in ys]}  mean={mean_auc:.4f}")

    if not any_data:
        raise SystemExit("No data found; check --runs-root.")

    ax.set_xticks(fold_indices)
    ax.set_xticklabels([f"Fold {i}" for i in fold_indices], fontsize=10)
    ax.set_xlabel("Cross-validation fold (subjects vary per fold)", fontsize=10)
    ax.set_ylabel("val_AUC", fontsize=10)
    ax.set_ylim(0.62, 0.83)
    ax.axhline(y=0.5, color="gray", linestyle=":", lw=1.0, alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(
        "Per-fold val_AUC: fold rank is constant across architectures\n"
        "(architecture-to-architecture spread at any fold ≪ fold-to-fold spread)",
        fontsize=11, pad=10,
    )

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Publication-quality figures for the fNIRS per-pair sweep paper.

Two subcommands, each producing a single focused figure:

  pretrained-vs-ablation
      Two val AUC curves on one axes: pretrained encoder vs random-init encoder
      using the same classifier head. Shows the representation-quality gap.

  model-holdouts
      One model's val AUC + gold holdout + salvageable holdout curves. Shows
      robustness to scan quality (child/adult discrimination on pristine vs
      high-motion data).

Output defaults to PDF (vectorized for print). Styled for single-column
or double-column figure inclusion — fonts sized for readability at 3–4 in.
figure width.

Examples:
    # Figure 1: the headline ablation
    python scripts/plot_paper_figures.py pretrained-vs-ablation \\
        --pretrained runs/fnirs_perpair_sweep/small_linear \\
        --ablation runs/fnirs_ablation_random/small_linear \\
        --title "Linear probe: pretrained vs random encoder" \\
        --output figures/linear_ablation.pdf

    # Figure 2: the quality-stratified robustness plot
    python scripts/plot_paper_figures.py model-holdouts \\
        --run runs/fnirs_perpair_sweep/small_lstm64 \\
        --title "Per-pair LSTM: validation vs holdout tiers" \\
        --output figures/lstm_holdouts.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Muted, colorblind-friendly palette (Okabe-Ito inspired)
COLOR_PRETRAINED = "#0072B2"   # blue
COLOR_ABLATION   = "#999999"   # gray
COLOR_VAL        = "#0072B2"   # blue
COLOR_GOLD       = "#E69F00"   # orange
COLOR_SALV       = "#D55E00"   # vermilion
COLOR_CHANCE     = "#888888"   # subtle gray


def _set_paper_style():
    """Apply matplotlib rcParams tuned for paper-figure inclusion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,     # embed TrueType fonts (editable in Illustrator)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def _load_history(run_dir: Path) -> dict:
    p = run_dir / "history.json"
    if not p.exists():
        raise FileNotFoundError(f"history.json not found in {run_dir}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: pretrained vs ablation (single classifier head)
# ---------------------------------------------------------------------------

def plot_pretrained_vs_ablation(
    pretrained_dir: Path,
    ablation_dir: Path,
    title: str,
    output: Path,
    metric: str = "val_aucs",
    ylabel: str = "Validation AUC",
    figsize: tuple[float, float] = (4.5, 3.2),
) -> None:
    """Two-line figure: pretrained vs ablation val AUC over epochs."""
    _set_paper_style()
    import matplotlib.pyplot as plt

    h_pre = _load_history(pretrained_dir)
    h_abl = _load_history(ablation_dir)

    y_pre = h_pre.get(metric, [])
    y_abl = h_abl.get(metric, [])
    if not y_pre or not y_abl:
        logger.error("Missing %s in one of the histories", metric)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(y_pre) + 1), y_pre,
            color=COLOR_PRETRAINED, label="Pretrained encoder")
    ax.plot(range(1, len(y_abl) + 1), y_abl,
            color=COLOR_ABLATION, linestyle="--",
            label="Random-init encoder (ablation)")

    # Chance line for AUC
    if "auc" in metric.lower():
        ax.axhline(0.5, color=COLOR_CHANCE, linestyle=":", linewidth=0.8,
                   label="Chance")
        ax.set_ylim(0.4, 1.0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.legend(loc="lower right")

    # Annotate final values
    best_pre = max(y_pre)
    best_abl = max(y_abl)
    ax.annotate(
        f"{best_pre:.3f}",
        xy=(y_pre.index(best_pre) + 1, best_pre),
        xytext=(5, 5), textcoords="offset points",
        fontsize=8, color=COLOR_PRETRAINED,
    )
    ax.annotate(
        f"{best_abl:.3f}",
        xy=(y_abl.index(best_abl) + 1, best_abl),
        xytext=(5, 5), textcoords="offset points",
        fontsize=8, color=COLOR_ABLATION,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    logger.info(
        "Saved → %s  (pretrained peak=%.3f, ablation peak=%.3f, Δ=%.3f)",
        output, best_pre, best_abl, best_pre - best_abl,
    )


# ---------------------------------------------------------------------------
# Figure 2: one model's val + holdout tiers
# ---------------------------------------------------------------------------

def plot_model_holdouts(
    run_dir: Path,
    title: str,
    output: Path,
    metric_key: str = "aucs",
    ylabel: str = "AUC",
    figsize: tuple[float, float] = (4.8, 3.4),
) -> None:
    """Val + gold + salvageable curves for a single run."""
    _set_paper_style()
    import matplotlib.pyplot as plt

    h = _load_history(run_dir)

    val = h.get(f"val_{metric_key}", [])
    gold = h.get(f"holdout_gold_{metric_key}", [])
    salv = h.get(f"holdout_salvageable_{metric_key}", [])
    if not val:
        logger.error("No val_%s in history — aborting", metric_key)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(val) + 1), val,
            color=COLOR_VAL, label="Validation (mixed tiers)")
    if gold:
        ax.plot(range(1, len(gold) + 1), gold,
                color=COLOR_GOLD, label="Holdout: gold (pristine)")
    if salv:
        ax.plot(range(1, len(salv) + 1), salv,
                color=COLOR_SALV, linestyle="--",
                label="Holdout: salvageable (high-motion)")

    if "auc" in metric_key.lower():
        ax.axhline(0.5, color=COLOR_CHANCE, linestyle=":", linewidth=0.8,
                   label="Chance")
        ax.set_ylim(0.4, 1.0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.legend(loc="lower right")

    # Annotate final gold value — the "best case" clinical number
    if gold:
        final_gold = gold[-1] if gold[-1] > 0 else max(gold)
        peak_gold = max(gold)
        ax.annotate(
            f"Gold peak: {peak_gold:.3f}",
            xy=(gold.index(peak_gold) + 1, peak_gold),
            xytext=(5, -12), textcoords="offset points",
            fontsize=8, color=COLOR_GOLD,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    peak_val = max(val)
    peak_gold = max(gold) if gold else None
    peak_salv = max(salv) if salv else None
    logger.info(
        "Saved → %s  (val peak=%.3f, gold peak=%s, salv peak=%s)",
        output, peak_val,
        f"{peak_gold:.3f}" if peak_gold is not None else "—",
        f"{peak_salv:.3f}" if peak_salv is not None else "—",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pretrained-vs-ablation")
    p1.add_argument("--pretrained", required=True, help="Path to pretrained run dir (has history.json)")
    p1.add_argument("--ablation", required=True, help="Path to ablation run dir (has history.json)")
    p1.add_argument("--title", required=True, help="Figure title")
    p1.add_argument("--output", required=True, help="Output file path (.pdf or .png)")
    p1.add_argument("--metric", default="val_aucs",
                    help="History key to plot (default: val_aucs). Use val_accs for accuracy.")
    p1.add_argument("--ylabel", default="Validation AUC",
                    help="Y-axis label (default: 'Validation AUC')")
    p1.add_argument("--width", type=float, default=4.5)
    p1.add_argument("--height", type=float, default=3.2)

    p2 = sub.add_parser("model-holdouts")
    p2.add_argument("--run", required=True, help="Path to run dir (has history.json)")
    p2.add_argument("--title", required=True, help="Figure title")
    p2.add_argument("--output", required=True, help="Output file path (.pdf or .png)")
    p2.add_argument("--metric", default="aucs",
                    help="Metric key suffix: 'aucs', 'accs', 'losses' (default: aucs)")
    p2.add_argument("--ylabel", default="AUC",
                    help="Y-axis label (default: 'AUC')")
    p2.add_argument("--width", type=float, default=4.8)
    p2.add_argument("--height", type=float, default=3.4)

    args = parser.parse_args()

    if args.cmd == "pretrained-vs-ablation":
        plot_pretrained_vs_ablation(
            pretrained_dir=Path(args.pretrained),
            ablation_dir=Path(args.ablation),
            title=args.title,
            output=Path(args.output),
            metric=args.metric,
            ylabel=args.ylabel,
            figsize=(args.width, args.height),
        )
    elif args.cmd == "model-holdouts":
        plot_model_holdouts(
            run_dir=Path(args.run),
            title=args.title,
            output=Path(args.output),
            metric_key=args.metric,
            ylabel=args.ylabel,
            figsize=(args.width, args.height),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

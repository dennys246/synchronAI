#!/usr/bin/env python3
"""
Reusable sweep/holdout/ablation comparison plots.

Discovers history.json files from run directories, generates publication-quality
comparison figures. Designed to be pointed at any sweep output directory —
swap out the path, get new plots.

Generates three figures:
  1. Sweep overview: all model variants on one figure (val AUC + val loss)
  2. Best model + holdout tiers: quality-stratified performance
  3. Pretrained vs ablation: proof that pretraining matters

Usage:
    # Full comparison (sweep + ablation):
    python scripts/plot_sweep_comparison.py \
        --sweep-dir runs/fnirs_perpair_sweep \
        --ablation-dir runs/fnirs_ablation_random \
        --output-dir runs/fnirs_perpair_sweep/plots

    # Just the sweep overview:
    python scripts/plot_sweep_comparison.py \
        --sweep-dir runs/fnirs_perpair_sweep

    # Custom runs (arbitrary list of history.json paths):
    python scripts/plot_sweep_comparison.py \
        --runs "lstm64=runs/sweep/small_lstm64" \
               "mlp32=runs/sweep/small_mlp32" \
        --output-dir runs/custom_comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- Style configuration ----
# Model sizes get colors, classifier types get line styles.
# Override or extend for different sweep structures.
SIZE_COLORS = {
    "micro": "#2196F3",   # blue
    "small": "#4CAF50",   # green
    "medium": "#FF9800",  # orange
    "large": "#F44336",   # red
}
CLASSIFIER_STYLES = {
    "linear": ":",
    "mlp32": "--",
    "lstm64": "-",
}
ABLATION_COLOR = "#9E9E9E"  # gray
GOLD_COLOR = "#FFD700"
SALVAGEABLE_COLOR = "#CD853F"


def load_history(run_dir: Path) -> dict | None:
    """Load history.json from a run directory. Returns None if missing."""
    path = run_dir / "history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def discover_sweep_runs(sweep_dir: Path) -> dict[str, dict]:
    """Discover all subdirectories with history.json files.

    Returns {run_name: history_dict} sorted by name.
    """
    runs = {}
    if not sweep_dir.exists():
        logger.warning("Sweep dir not found: %s", sweep_dir)
        return runs
    for sub in sorted(sweep_dir.iterdir()):
        if sub.is_dir():
            hist = load_history(sub)
            if hist:
                runs[sub.name] = hist
                logger.info("  loaded %s (%d epochs)", sub.name, len(hist.get("val_aucs", [])))
    return runs


def _parse_run_name(name: str) -> tuple[str, str]:
    """Split 'small_lstm64' into ('small', 'lstm64').

    Falls back to ('unknown', name) if the pattern doesn't match.
    """
    for size in ("micro", "small", "medium", "large"):
        if name.startswith(size + "_"):
            classifier = name[len(size) + 1:]
            return size, classifier
    return "unknown", name


def _style_for_run(name: str) -> dict:
    """Return matplotlib line kwargs for a run name."""
    size, classifier = _parse_run_name(name)
    return {
        "color": SIZE_COLORS.get(size, "#000000"),
        "linestyle": CLASSIFIER_STYLES.get(classifier, "-"),
        "linewidth": 1.8,
        "label": name,
    }


def _best_run(runs: dict[str, dict], metric: str = "val_aucs") -> tuple[str, dict]:
    """Return the (name, history) of the run with the highest peak metric."""
    best_name, best_val = None, -1
    for name, hist in runs.items():
        vals = hist.get(metric, [])
        if vals and max(vals) > best_val:
            best_val = max(vals)
            best_name = name
    return best_name, runs[best_name]


def plot_sweep_overview(runs: dict[str, dict], output_path: Path) -> None:
    """Figure 1: all sweep models on one 2-panel figure (val AUC + val loss)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_auc, ax_loss) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name, hist in runs.items():
        style = _style_for_run(name)
        epochs = range(1, len(hist.get("val_aucs", [])) + 1)
        if hist.get("val_aucs"):
            ax_auc.plot(epochs, hist["val_aucs"], **style)
        if hist.get("val_losses"):
            ax_loss.plot(epochs, hist["val_losses"], **style)

    ax_auc.set_ylabel("Validation AUC")
    ax_auc.set_ylim(0.4, 1.0)
    ax_auc.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_auc.legend(fontsize=7, ncol=3, loc="lower right")
    ax_auc.grid(alpha=0.2)
    ax_auc.set_title("Sweep overview — validation AUC")

    ax_loss.set_ylabel("Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend(fontsize=7, ncol=3, loc="upper right")
    ax_loss.grid(alpha=0.2)
    ax_loss.set_title("Sweep overview — validation loss")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sweep overview → %s", output_path)


def plot_holdout_comparison(
    best_name: str,
    best_hist: dict,
    ablation_hist: dict | None,
    output_path: Path,
) -> None:
    """Figure 2: best model's val AUC alongside holdout tiers + ablation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2  # AUC + accuracy
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 5))

    for ax, (metric_key, metric_label) in zip(axes, [
        ("aucs", "AUC"),
        ("accs", "Accuracy"),
    ]):
        epochs = range(1, len(best_hist.get(f"val_{metric_key}", [])) + 1)

        # Best model val
        if best_hist.get(f"val_{metric_key}"):
            ax.plot(epochs, best_hist[f"val_{metric_key}"],
                    color="#2196F3", linewidth=2, label=f"{best_name} (val)")

        # Gold holdout
        key = f"holdout_gold_{metric_key}"
        if best_hist.get(key):
            ax.plot(range(1, len(best_hist[key]) + 1), best_hist[key],
                    color=GOLD_COLOR, linewidth=2, linestyle="--",
                    label="Holdout [gold]")

        # Salvageable holdout
        key = f"holdout_salvageable_{metric_key}"
        if best_hist.get(key):
            ax.plot(range(1, len(best_hist[key]) + 1), best_hist[key],
                    color=SALVAGEABLE_COLOR, linewidth=2, linestyle="--",
                    label="Holdout [salvageable]")

        # Ablation (random init) — if provided
        if ablation_hist and ablation_hist.get(f"val_{metric_key}"):
            abl_epochs = range(1, len(ablation_hist[f"val_{metric_key}"]) + 1)
            ax.plot(abl_epochs, ablation_hist[f"val_{metric_key}"],
                    color=ABLATION_COLOR, linewidth=2, linestyle="-.",
                    label="Ablation (random init)")

        if metric_key == "aucs":
            ax.set_ylim(0.4, 1.0)
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        else:
            ax.set_ylim(0.0, 1.0)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Best model vs holdout tiers — {metric_label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(f"Best model: {best_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved holdout comparison → %s", output_path)


def plot_pretrained_vs_ablation(
    pretrained_runs: dict[str, dict],
    ablation_runs: dict[str, dict],
    output_path: Path,
) -> None:
    """Figure 3: pretrained models vs their random-init ablations (val AUC)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Pretrained runs
    for name, hist in pretrained_runs.items():
        style = _style_for_run(name)
        epochs = range(1, len(hist.get("val_aucs", [])) + 1)
        if hist.get("val_aucs"):
            ax.plot(epochs, hist["val_aucs"], **style)

    # Ablation runs (gray, dashed)
    for name, hist in ablation_runs.items():
        epochs = range(1, len(hist.get("val_aucs", [])) + 1)
        if hist.get("val_aucs"):
            ax.plot(epochs, hist["val_aucs"],
                    color=ABLATION_COLOR, linestyle="-.", linewidth=1.5,
                    label=f"ablation: {name}", alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title("Pretrained encoder vs random-init ablation")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved pretrained vs ablation → %s", output_path)


def plot_final_bar_chart(
    runs: dict[str, dict],
    ablation_runs: dict[str, dict],
    output_path: Path,
) -> None:
    """Bonus: bar chart of peak val AUC per model, with ablation baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect peak AUCs
    entries = []
    for name, hist in sorted(runs.items()):
        vals = hist.get("val_aucs", [])
        if vals:
            size, classifier = _parse_run_name(name)
            entries.append({"name": name, "peak_auc": max(vals), "type": "pretrained",
                            "color": SIZE_COLORS.get(size, "#000000")})

    for name, hist in sorted(ablation_runs.items()):
        vals = hist.get("val_aucs", [])
        if vals:
            entries.append({"name": f"abl:{name}", "peak_auc": max(vals),
                            "type": "ablation", "color": ABLATION_COLOR})

    if not entries:
        return

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(entries) * 0.8), 5))

    names = [e["name"] for e in entries]
    aucs = [e["peak_auc"] for e in entries]
    colors = [e["color"] for e in entries]

    bars = ax.bar(range(len(names)), aucs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peak validation AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title("Peak validation AUC — all models")
    ax.grid(axis="y", alpha=0.2)

    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bar chart → %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate sweep/holdout/ablation comparison plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sweep-dir", default=None,
        help="Directory containing sweep runs (each subdir has history.json). "
             "E.g. runs/fnirs_perpair_sweep",
    )
    parser.add_argument(
        "--ablation-dir", default=None,
        help="Directory containing ablation runs. "
             "E.g. runs/fnirs_ablation_random",
    )
    parser.add_argument(
        "--runs", nargs="*", default=[],
        help="Explicit runs as 'label=path' pairs. "
             "E.g. 'lstm64=runs/sweep/small_lstm64'",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for plots (default: sweep_dir/plots or ./plots)",
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"],
        help="Output format (default: png)",
    )
    args = parser.parse_args()

    # Collect runs
    sweep_runs: dict[str, dict] = {}
    ablation_runs: dict[str, dict] = {}

    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        logger.info("Discovering sweep runs in %s", sweep_dir)
        sweep_runs = discover_sweep_runs(sweep_dir)
        logger.info("Found %d sweep runs", len(sweep_runs))

    if args.ablation_dir:
        abl_dir = Path(args.ablation_dir)
        logger.info("Discovering ablation runs in %s", abl_dir)
        ablation_runs = discover_sweep_runs(abl_dir)
        logger.info("Found %d ablation runs", len(ablation_runs))

    # Explicit --runs override / supplement
    for run_spec in args.runs:
        if "=" in run_spec:
            label, path = run_spec.split("=", 1)
        else:
            label = Path(run_spec).name
            path = run_spec
        hist = load_history(Path(path))
        if hist:
            sweep_runs[label] = hist
            logger.info("  loaded explicit run: %s", label)
        else:
            logger.warning("  no history.json in %s", path)

    if not sweep_runs and not ablation_runs:
        logger.error("No runs found. Pass --sweep-dir, --ablation-dir, or --runs.")
        return 1

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.sweep_dir:
        out_dir = Path(args.sweep_dir) / "plots"
    else:
        out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format

    # Figure 1: sweep overview
    if sweep_runs:
        plot_sweep_overview(sweep_runs, out_dir / f"sweep_overview.{fmt}")

    # Figure 2: best model + holdout tiers + ablation
    if sweep_runs:
        best_name, best_hist = _best_run(sweep_runs)
        logger.info("Best sweep run: %s (peak val AUC %.4f)",
                     best_name, max(best_hist.get("val_aucs", [0])))

        # Find matching ablation if available
        best_size, best_cls = _parse_run_name(best_name)
        abl_hist = None
        for abl_name, abl_h in ablation_runs.items():
            if best_cls in abl_name or best_size in abl_name:
                abl_hist = abl_h
                logger.info("Matched ablation: %s", abl_name)
                break
        if abl_hist is None and ablation_runs:
            # Fall back to first ablation run
            abl_name = next(iter(ablation_runs))
            abl_hist = ablation_runs[abl_name]
            logger.info("Using ablation fallback: %s", abl_name)

        plot_holdout_comparison(best_name, best_hist, abl_hist,
                                out_dir / f"holdout_comparison.{fmt}")

    # Figure 3: pretrained vs ablation
    if sweep_runs and ablation_runs:
        plot_pretrained_vs_ablation(sweep_runs, ablation_runs,
                                    out_dir / f"pretrained_vs_ablation.{fmt}")

    # Bonus: bar chart of peak AUCs
    if sweep_runs:
        plot_final_bar_chart(sweep_runs, ablation_runs,
                             out_dir / f"peak_auc_bar.{fmt}")

    logger.info("All plots saved to %s/", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

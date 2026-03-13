#!/usr/bin/env python3
"""Compare results from a DINOv2 feature training sweep.

Reads history.json and config.json from each run directory and produces:
  1. A summary table printed to stdout
  2. A comparison plot saved as sweep_comparison.png

Usage:
    python scripts/compare_sweep_results.py --sweep-dir runs/dinov2_sweep
"""

import argparse
import json
import sys
from pathlib import Path


def load_run(run_dir: Path) -> dict | None:
    """Load history and config from a single run directory."""
    history_file = run_dir / "history.json"
    config_file = run_dir / "config.json"

    if not history_file.exists():
        return None

    with open(history_file) as f:
        history = json.load(f)

    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return {
        "name": run_dir.name,
        "history": history,
        "config": config,
    }


def print_summary_table(runs: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    # Header
    cols = [
        ("Run", 18),
        ("Temporal", 10),
        ("Hidden", 7),
        ("Drop", 5),
        ("LR", 8),
        ("WD", 8),
        ("Best AUC", 9),
        ("Best F1", 8),
        ("Best Ep", 8),
        ("Train L", 8),
        ("Val L", 8),
        ("Gap", 8),
    ]

    header = " | ".join(f"{name:<{width}}" for name, width in cols)
    separator = "-+-".join("-" * width for _, width in cols)

    print()
    print("=" * len(header))
    print("  DINOv2 Feature Training Sweep Results")
    print("=" * len(header))
    print(header)
    print(separator)

    # Sort by best AUC descending
    runs_sorted = sorted(runs, key=lambda r: r["history"].get("best_val_auc", 0), reverse=True)

    for run in runs_sorted:
        h = run["history"]
        c = run["config"]

        best_epoch = h.get("best_epoch", 0)
        best_auc = h.get("best_val_auc", 0)

        # Get metrics at best epoch
        best_f1 = h["val_f1s"][best_epoch] if best_epoch < len(h.get("val_f1s", [])) else 0

        # Final train/val loss (last epoch that ran)
        final_train = h["train_losses"][-1] if h.get("train_losses") else 0
        final_val = h["val_losses"][-1] if h.get("val_losses") else 0
        gap = final_val - final_train

        row_data = [
            (run["name"], 18),
            (c.get("temporal_aggregation", "?"), 10),
            (str(c.get("hidden_dim", "?")), 7),
            (str(c.get("dropout", "?")), 5),
            (f"{c.get('learning_rate', 0):.0e}", 8),
            (f"{c.get('weight_decay', 0):.0e}", 8),
            (f"{best_auc:.4f}", 9),
            (f"{best_f1:.4f}", 8),
            (str(best_epoch + 1), 8),
            (f"{final_train:.4f}", 8),
            (f"{final_val:.4f}", 8),
            (f"{gap:+.4f}", 8),
        ]

        row = " | ".join(f"{val:<{width}}" for val, width in row_data)
        print(row)

    print(separator)
    print(f"  {len(runs)} runs compared. Sorted by Best AUC (descending).")
    print()


def generate_comparison_plot(runs: list[dict], output_path: Path) -> None:
    """Generate a comparison plot with all runs overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, run in enumerate(runs):
        h = run["history"]
        color = colors[i % len(colors)]
        label = run["name"]
        epochs = range(1, len(h["train_losses"]) + 1)

        # Train loss
        axes[0, 0].plot(epochs, h["train_losses"], color=color, linestyle="-",
                        label=f"{label} (train)", linewidth=1.5, alpha=0.7)
        axes[0, 0].plot(epochs, h["val_losses"], color=color, linestyle="--",
                        label=f"{label} (val)", linewidth=1.5, alpha=0.7)

        # Val AUC
        axes[0, 1].plot(epochs, h["val_aucs"], color=color, label=label,
                        linewidth=2, marker="o", markersize=3)

        # Val F1
        axes[1, 0].plot(epochs, h["val_f1s"], color=color, label=label,
                        linewidth=2, marker="s", markersize=3)

        # Overfitting gap (val_loss - train_loss)
        gap = [v - t for t, v in zip(h["train_losses"], h["val_losses"])]
        axes[1, 1].plot(epochs, gap, color=color, label=label, linewidth=2)

    # Format axes
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Train (solid) / Val (dashed) Loss")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AUC")
    axes[0, 1].set_title("Validation AUC")
    axes[0, 1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].set_title("Validation F1")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Val Loss - Train Loss")
    axes[1, 1].set_title("Overfitting Gap (lower = better)")
    axes[1, 1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "DINOv2 Feature Training — Hyperparameter Sweep Comparison",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DINOv2 feature training sweep results"
    )
    parser.add_argument(
        "--sweep-dir", default="runs/dinov2_sweep",
        help="Directory containing sweep run subdirectories",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating the comparison plot",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        sys.exit(1)

    # Find all run directories with history.json
    runs = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if run_dir.is_dir():
            run = load_run(run_dir)
            if run is not None:
                runs.append(run)

    if not runs:
        print(f"No completed runs found in {sweep_dir}")
        sys.exit(1)

    print_summary_table(runs)

    if not args.no_plot:
        try:
            plot_path = sweep_dir / "sweep_comparison.png"
            generate_comparison_plot(runs, plot_path)
        except ImportError:
            print("matplotlib not available — skipping plot generation")
        except Exception as e:
            print(f"Warning: could not generate plot: {e}")


if __name__ == "__main__":
    main()

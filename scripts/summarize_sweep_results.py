#!/usr/bin/env python3
"""
Summarize all sweep runs under a directory into a CSV.

Walks every subdirectory with a history.json, extracts best val AUC, best
epoch, final metrics, and holdout-tier performance at the best epoch, and
writes a CSV next to the sweep root.

Usage:
    python scripts/summarize_sweep_results.py \
        --sweep-dir runs/fnirs_perpair_sweep

    # With ablation side-by-side:
    python scripts/summarize_sweep_results.py \
        --sweep-dir runs/fnirs_perpair_sweep \
        --ablation-dir runs/fnirs_ablation_random
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_run_name(name: str) -> tuple[str, str]:
    """Split 'small_lstm64' into ('small', 'lstm64')."""
    for size in ("micro", "small", "medium", "large"):
        if name.startswith(size + "_"):
            return size, name[len(size) + 1:]
    return "unknown", name


def summarize_run(run_dir: Path, tag: str = "pretrained") -> dict | None:
    """Pull best metrics from a single run's history.json."""
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None

    with open(hist_path) as f:
        hist = json.load(f)

    val_aucs = hist.get("val_aucs", [])
    if not val_aucs:
        logger.warning("  %s: no val_aucs — skipping", run_dir.name)
        return None

    best_idx = max(range(len(val_aucs)), key=lambda i: val_aucs[i])
    best_epoch = best_idx + 1  # epochs are 1-indexed
    size, classifier = _parse_run_name(run_dir.name)

    def at_best(key: str) -> float | None:
        series = hist.get(key, [])
        if series and best_idx < len(series):
            return round(float(series[best_idx]), 4)
        return None

    row = {
        "run_name": run_dir.name,
        "tag": tag,
        "model_size": size,
        "classifier": classifier,
        "n_epochs": len(val_aucs),
        "best_epoch": best_epoch,
        "best_val_auc": round(float(val_aucs[best_idx]), 4),
        "final_val_auc": round(float(val_aucs[-1]), 4),
        "best_val_loss": at_best("val_losses"),
        "best_val_acc": at_best("val_accs"),
        "best_val_f1": at_best("val_f1s"),
        "holdout_gold_auc": at_best("holdout_gold_aucs"),
        "holdout_gold_acc": at_best("holdout_gold_accs"),
        "holdout_salvageable_auc": at_best("holdout_salvageable_aucs"),
        "holdout_salvageable_acc": at_best("holdout_salvageable_accs"),
        "learning_rate_at_best": at_best("learning_rates"),
    }
    return row


def summarize_directory(base_dir: Path, tag: str = "pretrained") -> list[dict]:
    rows = []
    if not base_dir.exists():
        logger.warning("Not found: %s", base_dir)
        return rows
    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir():
            row = summarize_run(sub, tag=tag)
            if row:
                rows.append(row)
                logger.info(
                    "  %-30s best_auc=%.4f @ epoch %d (of %d)",
                    sub.name, row["best_val_auc"], row["best_epoch"], row["n_epochs"],
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True,
                        help="Directory containing sweep runs")
    parser.add_argument("--ablation-dir", default=None,
                        help="Optional ablation directory for side-by-side rows")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: sweep-dir/sweep_results.csv)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    logger.info("Summarizing sweep: %s", sweep_dir)
    rows = summarize_directory(sweep_dir, tag="pretrained")

    if args.ablation_dir:
        abl_dir = Path(args.ablation_dir)
        logger.info("Summarizing ablation: %s", abl_dir)
        rows += summarize_directory(abl_dir, tag="ablation")

    if not rows:
        logger.error("No runs found with history.json")
        return 1

    output_path = Path(args.output) if args.output else sweep_dir / "sweep_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Sort by best_val_auc descending for a leaderboard print
    rows_sorted = sorted(rows, key=lambda r: r["best_val_auc"], reverse=True)
    print()
    print(f"Wrote {len(rows)} rows to {output_path}")
    print()
    print("Leaderboard (by best val AUC):")
    print(f"{'Run':<32} {'Tag':<12} {'BestAUC':>8} {'Gold':>7} {'Salv':>7} {'Epoch':>6}")
    for r in rows_sorted:
        gold = f"{r['holdout_gold_auc']:.3f}" if r['holdout_gold_auc'] is not None else "  —  "
        salv = f"{r['holdout_salvageable_auc']:.3f}" if r['holdout_salvageable_auc'] is not None else "  —  "
        print(
            f"{r['run_name']:<32} {r['tag']:<12} {r['best_val_auc']:>8.4f} "
            f"{gold:>7} {salv:>7} {r['best_epoch']:>6}/{r['n_epochs']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

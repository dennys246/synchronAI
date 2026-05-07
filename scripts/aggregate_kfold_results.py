#!/usr/bin/env python3
"""Aggregate per-fold history.json files from a k-fold CV run.

Reports per-fold metrics + mean / std / 95% CI across folds. Used to tighten
the confidence interval on a single-fold point estimate (e.g., v3_h24's
0.8031 val_AUC) before declaring a result publishable.

Usage:
    python scripts/aggregate_kfold_results.py \\
        --cv-dir runs/multimodal_features/v3_h24_cv5

Expects subdirs named fold_0/, fold_1/, ... each containing history.json.
"""

import argparse
import json
import math
from pathlib import Path


def load_fold_metrics(fold_dir: Path) -> dict:
    h = json.load(open(fold_dir / "history.json"))
    return {
        "best_val_acc": max(h["val_accs"]),
        "best_val_acc_epoch": h["val_accs"].index(max(h["val_accs"])) + 1,
        "best_val_auc": max(h["val_aucs"]),
        "best_val_auc_epoch": h["val_aucs"].index(max(h["val_aucs"])) + 1,
        "best_val_loss": min(h["val_losses"]),
        "best_val_loss_epoch": h["val_losses"].index(min(h["val_losses"])) + 1,
        "n_epochs_run": len(h["val_accs"]),
    }


def summarize(values: list[float], label: str) -> None:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    print(f"  {label:<10}  per-fold: {[round(v, 4) for v in values]}")
    print(f"               mean={mean:.4f}  std={std:.4f}  SE={se:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cv-dir", required=True, help="Parent dir with fold_0/, fold_1/, ...")
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir)
    fold_dirs = sorted(p for p in cv_dir.glob("fold_*") if (p / "history.json").exists())
    if not fold_dirs:
        raise SystemExit(f"No fold_*/history.json under {cv_dir}")

    accs, aucs, losses = [], [], []
    print(f"=== {cv_dir} ({len(fold_dirs)} folds) ===")
    print()
    print(f"{'fold':<8} {'val_acc':>8} {'(ep)':>4}   {'val_auc':>8} {'(ep)':>4}   {'val_loss':>9} {'(ep)':>4}")
    print("-" * 70)
    for fd in fold_dirs:
        m = load_fold_metrics(fd)
        accs.append(m["best_val_acc"])
        aucs.append(m["best_val_auc"])
        losses.append(m["best_val_loss"])
        print(
            f"{fd.name:<8} {m['best_val_acc']:>8.4f} {m['best_val_acc_epoch']:>4}   "
            f"{m['best_val_auc']:>8.4f} {m['best_val_auc_epoch']:>4}   "
            f"{m['best_val_loss']:>9.4f} {m['best_val_loss_epoch']:>4}"
        )
    print()
    print("=== Summary across folds ===")
    summarize(accs, "val_acc")
    summarize(aucs, "val_auc")
    summarize(losses, "val_loss")


if __name__ == "__main__":
    main()

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
    losses, accs, aucs = h["val_losses"], h["val_accs"], h["val_aucs"]
    # One epoch per fold, chosen by the early-stop metric (min val_loss = the
    # checkpoint best.pt keeps). Taking max(val_accs)/max(val_aucs)/min(val_losses)
    # independently overestimates true performance AND mixes three different
    # epochs that describe no single deployed model — the exact selection bias
    # this aggregation exists to remove.
    ep = min(range(len(losses)), key=lambda i: losses[i])
    return {
        "epoch": ep + 1,
        "val_acc": accs[ep],
        "val_auc": aucs[ep],
        "val_loss": losses[ep],
        "n_epochs_run": len(accs),
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
    print("metrics reported at each fold's early-stop epoch (min val_loss)")
    print()
    print(f"{'fold':<8} {'epoch':>6}   {'val_acc':>8}   {'val_auc':>8}   {'val_loss':>9}")
    print("-" * 60)
    for fd in fold_dirs:
        m = load_fold_metrics(fd)
        accs.append(m["val_acc"])
        aucs.append(m["val_auc"])
        losses.append(m["val_loss"])
        print(
            f"{fd.name:<8} {m['epoch']:>6}   {m['val_acc']:>8.4f}   "
            f"{m['val_auc']:>8.4f}   {m['val_loss']:>9.4f}"
        )
    print()
    print("=== Summary across folds ===")
    summarize(accs, "val_acc")
    summarize(aucs, "val_auc")
    summarize(losses, "val_loss")


if __name__ == "__main__":
    main()

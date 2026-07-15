"""Aggregate the family_id CV hyperparameter/arch sweep into a ranked table.

Reads every <sweep_dir>/<config>_fold<k>/history.json and reports, per config,
the across-fold mean +/- SE of:
  - val_auc @ best-val_loss epoch  -- the checkpoint early-stopping actually
    selects (best.pt = min val_loss), so this is the honest deployed number.
  - peak val_auc                   -- reference only; peaks at ~epoch 1 here
    (known warmup artifact), so it over-reads; shown for context.

Ranks by the deployed metric. Flags configs with missing folds.

Usage:
  python scripts/summarize_hp_sweep.py runs/multimodal_features/hpsweep_familysplit
"""

from __future__ import annotations

import json
import math
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _auc(y: list[int], p: list[float]) -> float | None:
    """Rank-based AUC with tie handling. None if a single class."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(p):
        j = i
        while j < len(p) and p[order[j]] == p[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    npos = sum(y)
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    sp = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (sp - npos * (npos + 1) / 2) / (npos * nneg)


def within_recording_auc(root: Path, config: str) -> tuple[float | None, float | None]:
    """Pool a config's fold val_predictions.csv, return (pooled_auc, within_rec_auc).

    within_rec = clip-count-weighted mean of per-recording AUC (ranks clips only
    against clips from the same video_path). The pooled-vs-within gap is the share
    of AUC coming from between-recording base-rate/identity, not moment-to-moment
    synchrony. Returns (None, None) if no prediction files exist.
    """
    by_rec: dict[str, dict] = defaultdict(lambda: {"y": [], "p": []})
    ally: list[int] = []
    allp: list[float] = []
    files = sorted(root.glob(f"{config}_fold*/val_predictions.csv"))
    if not files:
        return None, None
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    yt = int(float(r["y_true"]))
                    yp = float(r["y_prob"])
                except (KeyError, ValueError):
                    continue
                vp = r.get("video_path", "")
                by_rec[vp]["y"].append(yt)
                by_rec[vp]["p"].append(yp)
                ally.append(yt)
                allp.append(yp)
    pooled = _auc(ally, allp)
    per = [(_auc(v["y"], v["p"]), len(v["y"])) for v in by_rec.values()]
    per = [(a, n) for a, n in per if a is not None]
    within = sum(a * n for a, n in per) / sum(n for _, n in per) if per else None
    return pooled, within


def _series(h: dict, *names: str) -> list:
    for n in names:
        v = h.get(n)
        if v:
            return v
    return []


def fold_metrics(hist_path: Path) -> dict | None:
    try:
        h = json.loads(hist_path.read_text())
    except Exception:
        return None
    va = _series(h, "val_auc", "val_aucs")
    vl = _series(h, "val_loss", "val_losses")
    if not va:
        return None
    peak = max(va)
    # val_auc at the epoch with lowest val_loss (the checkpoint best.pt keeps)
    if vl:
        best_ep = min(range(len(vl)), key=lambda i: vl[i])
        deployed = va[best_ep] if best_ep < len(va) else va[-1]
    else:
        deployed = peak
    return {"deployed": deployed, "peak": peak, "epochs": len(va)}


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize_hp_sweep.py <sweep_dir>")
    root = Path(sys.argv[1])
    by_config: dict[str, list[dict]] = defaultdict(list)
    for hp in sorted(root.glob("*/history.json")):
        name = hp.parent.name
        config = name.rsplit("_fold", 1)[0] if "_fold" in name else name
        m = fold_metrics(hp)
        if m:
            by_config[config].append(m)

    if not by_config:
        raise SystemExit("no history.json files found under " + str(root))

    rows = []
    for config, folds in by_config.items():
        dep_m, dep_se = mean_se([f["deployed"] for f in folds])
        pk_m, _ = mean_se([f["peak"] for f in folds])
        pooled, within = within_recording_auc(root, config)
        rows.append((config, dep_m, dep_se, pk_m, len(folds), pooled, within))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"\n{'config':<24} {'deployed AUC':<20} {'pooled':<8} {'within-rec':<11} folds")
    print("-" * 74)
    for config, dep_m, dep_se, pk_m, nf, pooled, within in rows:
        flag = "" if nf == 5 else f"  <-- {nf}/5"
        ps = f"{pooled:.4f}" if pooled is not None else "  -  "
        ws = f"{within:.4f}" if within is not None else "  -  "
        print(f"{config:<24} {dep_m:.4f} +/- {dep_se:.4f}   {ps:<8} {ws:<11} {nf}{flag}")
    print("\ndeployed AUC = val_auc at min-val_loss epoch, mean +/- SE across folds.")
    print("pooled/within-rec AUC pooled over folds' val_predictions.csv (out-of-fold).")
    print("within-rec = clip-weighted per-recording AUC (dyad identity removed): the honest")
    print("moment-to-moment synchrony number. pooled - within-rec = between-recording")
    print("base-rate/identity share (the same confound the family leakage exploited maximally).")


if __name__ == "__main__":
    main()

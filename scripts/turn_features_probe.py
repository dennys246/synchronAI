#!/usr/bin/env python3
"""
Probe: does speaker-attributed turn structure predict dyadic synchrony?

This is a CEILING test, not a model. The turn features come from HUMAN verbal
coding (P-CAT R01 `ParentVerbal`/`ChildVerbal` tiers), so speaker attribution is
perfect. If turn structure carries no signal here, no automatic diarizer can do
better, and the pyannote fine-tuning effort that `upgrade_plan.md` §3.2 calls
mandatory can be skipped.

Deliberately low capacity: L2 logistic regression on the flattened
(2*window+1, 9) window. At 15 subjects anything larger would fit subject
identity rather than turn structure.

Evaluation is leave-one-SUBJECT-out. Per-file synchronous rates in this cohort
span 1.8%-76.4%, so k-fold would be a fold-composition lottery — this project
has already had a run of architecture claims invalidated by exactly that.

Caveat the caller must not forget: there is no matched video-only baseline for
R01 (its DINOv2/WavLM features are not extracted). This measures signal against
chance and against a within-subject label permutation null, NOT against video.

Usage:
    python scripts/turn_features_probe.py \
        --feature-dir data/turntaking_features_r01 \
        --labels-file data/labels_r01pcat.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC. NaN when only one class is present."""
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float, iters: int, lr: float) -> np.ndarray:
    """Plain L2 logistic regression by gradient descent. Bias is not penalised."""
    w = np.zeros(X.shape[1] + 1)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    # class weights so a 1.8%-synchronous subject does not collapse to the prior
    n1, n0 = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    sw = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (sw * (p - y)) / len(y)
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.hstack([X, np.ones((len(X), 1))]) @ w, -30, 30)))


def load(feature_dir: Path, labels_file: Path):
    meta = json.load(open(feature_dir / "features_meta.json"))
    shape = tuple(meta["shape"])
    packed = np.memmap(feature_dir / "features_packed.bin", dtype=meta["dtype"],
                       mode="r", shape=shape)
    idx = {}
    for r in csv.DictReader(open(feature_dir / "feature_index.csv")):
        if r["video_path"]:
            idx[(r["video_path"], int(r["second"]))] = int(r["row_idx"])

    rows, ys, subs = [], [], []
    n_lab = n_miss = 0
    for r in csv.DictReader(open(labels_file)):
        n_lab += 1
        k = (r["video_path"], int(r["second"]))
        if k not in idx:
            n_miss += 1
            continue
        rows.append(idx[k])
        ys.append(int(r["label"]))
        subs.append(r["subject_id"])
    if not rows:
        raise SystemExit("ERROR: no labelled second joined to a feature row.")
    X = np.asarray(packed[rows], dtype=np.float64).reshape(len(rows), -1)
    return X, np.asarray(ys), np.asarray(subs), meta, n_lab, n_miss


def loso(X, y, subs, l2, iters, lr, rng=None, shuffle="within"):
    """Leave-one-subject-out. Returns per-subject AUCs and pooled held-out scores.

    Each fold's AUC is computed inside ONE held-out subject, so it measures
    moment-to-moment discrimination, not the dyad-identity channel that inflates
    pooled AUCs elsewhere in this project.

    shuffle="global" is the null for "features carry no information about
    labels". shuffle="within" permutes inside each training subject, which
    preserves that subject's base rate — it is NOT a null for that hypothesis,
    because a model can still learn a valid feature->base-rate mapping from
    between-subject variation. Report it as a training-regime comparison only.
    """
    out, pooled_s, pooled_y = [], [], []
    for s in sorted(set(subs)):
        te = subs == s
        tr = ~te
        yt = y[tr].copy()
        if rng is not None:
            if shuffle == "global":
                yt = rng.permutation(yt)
            else:
                for t in sorted(set(subs[tr])):
                    m = subs[tr] == t
                    yt[m] = rng.permutation(yt[m])
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        w = fit_logreg((X[tr] - mu) / sd, yt, l2, iters, lr)
        p = predict((X[te] - mu) / sd, w)
        out.append((s, int(te.sum()), float(y[te].mean()), auc(y[te], p)))
        pooled_s.append(p)
        pooled_y.append(y[te])
    return out, np.concatenate(pooled_s), np.concatenate(pooled_y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--labels-file", required=True)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=20, help="Permutation-null repeats.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, subs, meta, n_lab, n_miss = load(Path(args.feature_dir), Path(args.labels_file))
    print(f"joined {len(y):,}/{n_lab:,} labelled seconds "
          f"({n_miss:,} had no turn features) across {len(set(subs))} subjects")
    print(f"features: {X.shape[1]} dims = {meta['shape'][1]} sec x {meta['shape'][2]} channels")
    print(f"overall synchronous rate: {y.mean():.3f}\n")

    res, ps, py = loso(X, y, subs, args.l2, args.iters, args.lr)
    print(f"{'subject':14s} {'n':>6s} {'sync%':>6s} {'AUC':>7s}")
    for s, n, bal, a in res:
        print(f"{s:14s} {n:6d} {bal:6.3f} {a:7.3f}" + ("   (single-class, AUC undefined)" if a != a else ""))

    aucs = np.array([a for *_, a in res], dtype=float)
    ok = aucs[~np.isnan(aucs)]
    print(f"\nper-subject mean AUC : {ok.mean():.4f} +/- {ok.std():.4f}  (n={len(ok)} evaluable)")
    print(f"pooled held-out AUC  : {auc(py, ps):.4f}")

    rng = np.random.default_rng(args.seed)
    nulls = {}
    for mode in ("global", "within"):
        vals = []
        for _ in range(args.n_perm):
            r, _, _ = loso(X, y, subs, args.l2, args.iters, args.lr, rng=rng, shuffle=mode)
            vals.append(np.nanmean(np.array([v for *_, v in r], dtype=float)))
        nulls[mode] = np.array(vals)

    g = nulls["global"]
    print(f"\nglobal-shuffle NULL  : {g.mean():.4f} +/- {g.std():.4f}   "
          f"<- the null for 'features are uninformative'")
    print(f"  empirical p        : {(np.sum(g >= ok.mean()) + 1) / (len(g) + 1):.3f}")
    w = nulls["within"]
    print(f"within-subject shuffle: {w.mean():.4f} +/- {w.std():.4f}   "
          f"<- NOT a null; training on subject-level base rates only")
    print(f"  i.e. {100 * (w.mean() - 0.5) / (ok.mean() - 0.5):.0f}% of the effect is "
          f"reachable without any second-level label pairing in training")
    return 0


if __name__ == "__main__":
    sys.exit(main())

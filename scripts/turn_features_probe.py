#!/usr/bin/env python3
"""
Probe: which modality predicts dyadic synchrony, under one identical protocol?

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

Why not `train_multimodal_from_features.py`: its branches are ~100k-parameter
LSTMs. With 15 subjects that capacity fits subject identity rather than
synchrony, and the comparison stops meaning anything. Everything here is one
L2 logistic regression on pooled features, so video, turn and their combination
are judged at the same capacity and by the same LOSO folds.

Video and audio features are per-window .pt files and need torch; turn features
are packed numpy and do not. torch is imported lazily so the turn-only path
still runs anywhere.

Usage:
    # turn features only (no torch needed)
    python scripts/turn_features_probe.py \
        --turn-dir data/turntaking_features_r01 \
        --labels-file data/labels_r01pcat.csv

    # three-way comparison on the same subjects and folds
    python scripts/turn_features_probe.py \
        --labels-file data/labels_r01pcat.csv \
        --turn-dir  data/turntaking_features_r01 \
        --video-dir data/dinov2_features_meanpatch_r01pcat \
        --audio-dir data/wavlm_baseplus_features_r01pcat
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


def load_pt_dir(feature_dir: Path, keys: list[tuple[str, int]]):
    """Mean-pool each window's .pt over frames -> (n, D). Frame-mean keeps the
    dimensionality at the encoder width instead of frames x width, which at this
    sample size is the difference between a fit and a memorisation."""
    import torch  # lazy: only video/audio need it
    idx = {}
    for r in csv.DictReader(open(feature_dir / "feature_index.csv")):
        idx[(r["video_path"], int(float(r["second"])))] = r["feature_file"]
    out, miss = [], 0
    for k in keys:
        f = idx.get(k)
        if f is None:
            out.append(None)
            miss += 1
            continue
        x = torch.load(feature_dir / "features" / f, map_location="cpu",
                       weights_only=True).detach().float().numpy()
        out.append(x.reshape(x.shape[0], -1).mean(0) if x.ndim >= 2 else x.ravel())
    return out, miss


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
    keys = [(r["video_path"], int(r["second"]))
            for r in csv.DictReader(open(labels_file))
            if (r["video_path"], int(r["second"])) in idx]
    return X, np.asarray(ys), np.asarray(subs), meta, n_lab, n_miss, keys


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
    ap.add_argument("--turn-dir", "--feature-dir", dest="turn_dir", required=True)
    ap.add_argument("--labels-file", required=True)
    ap.add_argument("--video-dir", help="DINOv2 dir; adds a video-only and a combined arm.")
    ap.add_argument("--audio-dir", help="WavLM dir; adds an audio-only arm.")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=20, help="Permutation-null repeats.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, subs, meta, n_lab, n_miss, keys = load(Path(args.turn_dir), Path(args.labels_file))
    print(f"joined {len(y):,}/{n_lab:,} labelled seconds "
          f"({n_miss:,} had no turn features) across {len(set(subs))} subjects")
    print(f"features: {X.shape[1]} dims = {meta['shape'][1]} sec x {meta['shape'][2]} channels")
    print(f"overall synchronous rate: {y.mean():.3f}\n")

    arms = {"turn": X}
    for name, d in (("video", args.video_dir), ("audio", args.audio_dir)):
        if not d:
            continue
        vecs, _ = load_pt_dir(Path(d), keys)
        keep = [i for i, v in enumerate(vecs) if v is not None]
        if len(keep) != len(vecs):
            print(f"NOTE: {name}: {len(vecs) - len(keep)} of {len(vecs)} labelled seconds "
                  f"had no feature row; dropping from EVERY arm so all arms are "
                  f"scored on identical rows.")
            X = X[keep]; y = y[keep]; subs = subs[keep]
            keys = [keys[i] for i in keep]
            arms = {k: v[keep] for k, v in arms.items()}
            vecs = [vecs[i] for i in keep]
        arms[name] = np.asarray(vecs, dtype=np.float64)
    if "video" in arms:
        arms["video+turn"] = np.hstack([arms["video"], arms["turn"]])
    if "video" in arms and "audio" in arms:
        arms["video+audio+turn"] = np.hstack([arms["video"], arms["audio"], arms["turn"]])

    print(f"{'arm':20s} {'dims':>6s} {'mean AUC':>9s} {'sd':>7s} {'pooled':>8s}")
    summary = {}
    for name, Xa in arms.items():
        r, ps, py = loso(Xa, y, subs, args.l2, args.iters, args.lr)
        a = np.array([v for *_, v in r], dtype=float)
        summary[name] = (a[~np.isnan(a)], r)
        print(f"{name:20s} {Xa.shape[1]:6d} {summary[name][0].mean():9.4f} "
              f"{summary[name][0].std():7.4f} {auc(py, ps):8.4f}")

    ok, res = summary["turn"]
    print(f"\nper-subject detail, turn arm:")
    print(f"{'subject':14s} {'n':>6s} {'sync%':>6s} {'AUC':>7s}")
    for s, n, bal, a in res:
        print(f"{s:14s} {n:6d} {bal:6.3f} {a:7.3f}")

    if "video+turn" in summary:
        d = summary["video+turn"][0] - summary["video"][0]
        print(f"\nper-subject delta (video+turn) - video : {d.mean():+.4f} +/- {d.std():.4f}"
              f"   ({int((d > 0).sum())}/{len(d)} subjects improved)")
        print("  positive => turn structure is ADDITIVE to video; "
              "~0 => redundant, as WavLM and prosody were")

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

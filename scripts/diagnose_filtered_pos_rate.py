#!/usr/bin/env python3
"""Print per-fold class-distribution shift caused by the bogus-segment filter.

The fold-0 filtered v6 run showed val collapse (val_AUC 0.581). One of
the three competing hypotheses is class-distribution shift: filtering
changed train pos_weight from 0.804 → 1.492. If val pos_rate is substantially
different from train pos_rate after filtering, the model trained against
wrong class-balance will misclassify val systematically.

This script replicates the trainer's filter + subject_kfold_split logic on
labels.csv + fallback CSV (no torch, no features, no cluster needed), and
prints per-fold train/val class composition for both filtered and unfiltered
cases.

Usage:
    python scripts/diagnose_filtered_pos_rate.py \\
        --labels data/labels.csv \\
        --fallback data/perperson_video_features_conf005/fallback_per_segment.csv \\
        --num-folds 5 \\
        --seed 42 \\
        --max-fallback-frames 11
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def kfold_split(entries, num_folds, fold_idx, seed):
    """Mirror scripts/train_multimodal_from_features.py:subject_kfold_split."""
    by_subj = defaultdict(list)
    for e in entries:
        by_subj[e["subject_id"]].append(e)
    subjects = sorted(by_subj.keys())
    # numpy default_rng(seed).shuffle uses Mersenne Twister-like internals;
    # Python's random.Random(seed) doesn't match exactly, but the structural
    # answer (which subjects land together) is what we want, and we'll print
    # the chosen val subject set explicitly so you can compare to logs.
    rng = random.Random(seed)
    rng.shuffle(subjects)
    fold_size = len(subjects) // num_folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(subjects)
    val_subjects = set(subjects[val_start:val_end])
    train, val = [], []
    for s, group in by_subj.items():
        (val if s in val_subjects else train).extend(group)
    return train, val, sorted(val_subjects)


def class_stats(entries):
    if not entries:
        return 0, 0, 0.0
    pos = sum(1 for e in entries if int(float(e["label"])) == 1)
    neg = len(entries) - pos
    return pos, neg, pos / len(entries) if entries else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fallback", type=Path, required=True)
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fallback-frames", type=int, default=11)
    args = ap.parse_args()

    # Load labels
    with open(args.labels) as f:
        entries = list(csv.DictReader(f))
    n_in = len(entries)

    # Build bogus set
    bogus = set()
    with open(args.fallback) as f:
        for r in csv.DictReader(f):
            if int(r["n_fallback_frames"]) > args.max_fallback_frames:
                bogus.add((r["video_path"], int(float(r["second"]))))

    # Filtered entries
    filt_entries = [
        e for e in entries
        if (e["video_path"], int(e["second"])) not in bogus
    ]
    print(f"Labels: {n_in:,} total → {len(filt_entries):,} after filter "
          f"({(n_in - len(filt_entries)) / n_in * 100:.1f}% dropped)\n")

    for label, ent in [("UNFILTERED", entries), ("FILTERED  ", filt_entries)]:
        pos, neg, rate = class_stats(ent)
        pw = (neg / pos) if pos > 0 else float("inf")
        print(f"{label}  total: n={len(ent):>6,d}  pos={pos:>6,d}  neg={neg:>6,d}  "
              f"pos_rate={rate*100:.1f}%  pos_weight={pw:.3f}")
    print()

    # Per-fold breakdown for both filtered and unfiltered
    for label, ent in [("UNFILTERED", entries), ("FILTERED  ", filt_entries)]:
        print(f"=== {label.strip()} — per-fold class distribution ===")
        print(f"{'fold':>4s}  {'train_n':>7s}  {'train_pos%':>11s}  {'train_pw':>9s}  "
              f"{'val_n':>6s}  {'val_pos%':>9s}  {'val_pw':>9s}  shift")
        for k in range(args.num_folds):
            tr, va, val_subjects = kfold_split(ent, args.num_folds, k, args.seed)
            tp, tn, tr_rate = class_stats(tr)
            vp, vn, va_rate = class_stats(va)
            t_pw = (tn / tp) if tp > 0 else float("inf")
            v_pw = (vn / vp) if vp > 0 else float("inf")
            shift = abs(tr_rate - va_rate) * 100
            shift_mark = " <-- BIG SHIFT" if shift > 10 else ""
            print(f"{k:>4d}  {len(tr):>7,d}  {tr_rate*100:>10.1f}%  {t_pw:>9.3f}  "
                  f"{len(va):>6,d}  {va_rate*100:>8.1f}%  {v_pw:>9.3f}  Δ={shift:>4.1f}pt{shift_mark}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Score diarizer output against the R01 human verbal VAD, on the per-second grid.

Reports DER and its components, plus the error STRUCTURE — burstiness — because
that is exactly what the simulated curve in docs/results/turn_structure_probe.md
could not model. That simulation injected i.i.d. per-second errors; if real
errors arrive in long runs instead, the flat degradation curve is optimistic and
the measured AUC will fall below it.

Speaker identity is irrelevant here (established: swapping parent/child for whole
records barely moved AUC), so both cluster-to-person mappings are scored and the
better one is taken per record — which is what a downstream user of the labels
would effectively get for free.

Scoring is restricted to seconds the humans actually coded (the union of Trial
spans), since outside them there is no ground truth to compare against.

Usage:
    python scripts/score_diarization.py \
        --diar-dir data/diarization_r01 \
        --verbal-csv verbal_pcat_r01_07-27-2026.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_human(csv_path: Path):
    """Per-record (parent, child, covered) per-second boolean timelines."""
    speech = defaultdict(lambda: {"ParentVerbal": [], "ChildVerbal": []})
    trials = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        try:
            on, off = float(r["onset"]) / 1000.0, float(r["offset"]) / 1000.0
        except ValueError:
            continue
        if off <= on:
            continue
        rid, col, content = r["record_id"].strip(), r["column_name"], r["content"].strip()
        if col == "Trial":
            trials[rid].append((on, off))
        elif col in speech[rid] and content == "y":
            speech[rid][col].append((on, off))
    out = {}
    for rid, tr in trials.items():
        if not tr:
            continue
        n = int(np.ceil(max(b for _, b in tr)))
        def fill(iv):
            a = np.zeros(n, dtype=bool)
            for s, e in iv:
                a[int(np.floor(s)):min(n, int(np.ceil(e)))] = True
            return a
        out[rid] = (fill(speech[rid]["ParentVerbal"]), fill(speech[rid]["ChildVerbal"]),
                    fill(tr))
    return out


def runs(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs — the burstiness measure."""
    out, c = [], 0
    for v in mask:
        if v:
            c += 1
        elif c:
            out.append(c); c = 0
    if c:
        out.append(c)
    return out


def score(hp, hc, cov, dia):
    n = min(len(cov), len(dia))
    hp, hc, cov, dia = hp[:n], hc[:n], cov[:n], dia[:n]
    best = None
    for perm in ((0, 1), (1, 0)):
        dp, dc = dia[:, perm[0]], dia[:, perm[1]]
        hs, ds = (hp | hc) & cov, (dp | dc) & cov
        miss = int((hs & ~ds).sum())
        fa = int((~hs & ds).sum())
        both = hs & ds
        conf = int((both & ((hp & ~dp) | (hc & ~dc))).sum())
        tot = max(int(hs.sum()), 1)
        der = (miss + fa + conf) / tot
        if best is None or der < best[0]:
            err = (hs & ~ds) | (~hs & ds) | (both & ((hp & ~dp) | (hc & ~dc)))
            best = (der, miss / tot, fa / tot, conf / tot, tot, err & cov)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diar-dir", required=True)
    ap.add_argument("--verbal-csv", required=True)
    args = ap.parse_args()

    human = load_human(Path(args.verbal_csv))
    files = sorted(Path(args.diar_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"ERROR: no .npz under {args.diar_dir}")

    print(f"{'record':>8s} {'coded s':>8s} {'DER':>7s} {'miss':>7s} {'FA':>7s} "
          f"{'conf':>7s} {'errRun':>7s}")
    ders, rows, allruns = [], 0, []
    for f in files:
        rid = f.stem
        if rid not in human:
            print(f"{rid:>8s}   no human coding — skipped")
            continue
        d = np.load(f, allow_pickle=True)["active"]
        der, miss, fa, conf, tot, err = score(*human[rid], d)
        rl = runs(err)
        allruns += rl
        med = float(np.median(rl)) if rl else 0.0
        ders.append(der); rows += 1
        print(f"{rid:>8s} {tot:8d} {der:7.3f} {miss:7.3f} {fa:7.3f} {conf:7.3f} {med:7.1f}")

    if not ders:
        raise SystemExit("ERROR: no records scored.")
    ders = np.array(ders)
    print(f"\n{rows} records   DER mean {ders.mean():.3f} +/- {ders.std():.3f}  "
          f"median {np.median(ders):.3f}  range {ders.min():.3f}-{ders.max():.3f}")
    if allruns:
        a = np.array(allruns)
        print(f"error-run lengths: median {np.median(a):.1f}s  p90 {np.percentile(a, 90):.1f}s  "
              f"max {a.max()}s  ({100 * (a >= 3).mean():.0f}% of runs >= 3s)")
        print("  i.i.d. errors would give runs of ~1s. Long runs mean the simulated")
        print("  DER curve is optimistic — verify by rebuilding turn features from")
        print("  this diarizer's output and re-running turn_features_probe.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

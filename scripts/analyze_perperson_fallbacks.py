#!/usr/bin/env python3
"""Deep-dive on per-person feature fallback rates.

Reads the diagnostic CSVs produced by build_perperson_video_features.py
and produces a structured analysis: per-subject rates, worst-offender
subjects/sessions, fallback distribution, and (if available) per-activity
breakdown derived from the video_path stem.

Usage:
    python scripts/analyze_perperson_fallbacks.py \\
        --output-dir data/perperson_video_features_conf005
"""

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _activity_from_path(video_path: str) -> str:
    """CARE filenames look like 50001_V0_DB-DOS.mp4 — pull out the activity tag."""
    stem = Path(video_path).stem
    m = re.match(r"^\d+_[A-Za-z0-9]+_(.+)$", stem)
    return m.group(1) if m else stem


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="The per-person feature output dir (contains the CSVs)")
    args = ap.parse_args()

    per_seg_path = args.output_dir / "fallback_per_segment.csv"
    per_sub_path = args.output_dir / "fallback_per_subject.csv"
    if not per_seg_path.exists():
        logger.error(f"Missing {per_seg_path}")
        return 1

    rows = list(csv.DictReader(open(per_seg_path)))
    logger.info(f"Loaded {len(rows)} per-(segment, slot) diagnostic rows.")

    # ----- Global rate -----
    total_frames = 0
    fb_slot = [0, 0]
    for r in rows:
        slot = int(r["slot"])
        if slot == 0:
            total_frames += int(r["n_total_frames"])
        fb_slot[slot] += int(r["n_fallback_frames"])
    print()
    print("=== Global fallback rate ===")
    print(f"  total frames:       {total_frames:,}")
    print(f"  slot 0 fallbacks:   {fb_slot[0]:,}  ({fb_slot[0]/total_frames*100:.1f}%)")
    print(f"  slot 1 fallbacks:   {fb_slot[1]:,}  ({fb_slot[1]/total_frames*100:.1f}%)")

    # ----- Per-subject worst offenders -----
    if per_sub_path.exists():
        sub_rows = list(csv.DictReader(open(per_sub_path)))
        # Use combined (slot0 + slot1) rate as the badness score.
        def combined(r):
            return (float(r["fallback_rate_slot0"]) + float(r["fallback_rate_slot1"])) / 2
        sub_rows.sort(key=combined, reverse=True)
        print()
        print("=== Worst 10 (subject, session) by mean(slot0,slot1) fallback rate ===")
        print(f"  {'subject':>10s} {'session':>8s} {'n_seg':>6s} {'slot0%':>8s} {'slot1%':>8s}")
        for r in sub_rows[:10]:
            print(f"  {r['subject_id']:>10s} {r['session']:>8s} {int(r['n_segments']):>6d} "
                  f"{float(r['fallback_rate_slot0'])*100:>8.1f} "
                  f"{float(r['fallback_rate_slot1'])*100:>8.1f}")
        print()
        print("=== Best 10 (subject, session) ===")
        for r in sub_rows[-10:]:
            print(f"  {r['subject_id']:>10s} {r['session']:>8s} {int(r['n_segments']):>6d} "
                  f"{float(r['fallback_rate_slot0'])*100:>8.1f} "
                  f"{float(r['fallback_rate_slot1'])*100:>8.1f}")
        # Distribution of dyad-level (mean across both slots) fallback rates
        rates = [combined(r) for r in sub_rows]
        rates.sort()
        n = len(rates)
        def pct(p):
            return rates[min(n - 1, int(p / 100 * n))]
        print()
        print("=== Distribution of per-dyad mean fallback rate ===")
        print(f"  n_dyads:      {n}")
        print(f"  min:          {rates[0]*100:>5.1f}%")
        print(f"  25th pct:     {pct(25)*100:>5.1f}%")
        print(f"  median:       {pct(50)*100:>5.1f}%")
        print(f"  75th pct:     {pct(75)*100:>5.1f}%")
        print(f"  max:          {rates[-1]*100:>5.1f}%")
        clean = sum(1 for r in rates if r < 0.10)
        moderate = sum(1 for r in rates if 0.10 <= r < 0.30)
        bad = sum(1 for r in rates if r >= 0.30)
        print()
        print("=== Dyad bucketing ===")
        print(f"  <10%  fallback (usable):  {clean:>3d} dyads")
        print(f"  10-30% fallback (degraded): {moderate:>3d} dyads")
        print(f"  >30%  fallback (broken):    {bad:>3d} dyads")

    # ----- Per-activity breakdown -----
    act_agg = defaultdict(lambda: {"total": 0, "fb0": 0, "fb1": 0})
    for r in rows:
        slot = int(r["slot"])
        act = _activity_from_path(r["video_path"])
        if slot == 0:
            act_agg[act]["total"] += int(r["n_total_frames"])
        act_agg[act][f"fb{slot}"] += int(r["n_fallback_frames"])
    print()
    print("=== Per-activity (filename suffix) fallback rate ===")
    print(f"  {'activity':>20s} {'n_total':>10s} {'slot0%':>8s} {'slot1%':>8s}")
    for act, a in sorted(act_agg.items(), key=lambda kv: -kv[1]["total"]):
        if a["total"] == 0:
            continue
        print(f"  {act[:20]:>20s} {a['total']:>10,d} "
              f"{a['fb0']/a['total']*100:>8.1f} "
              f"{a['fb1']/a['total']*100:>8.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

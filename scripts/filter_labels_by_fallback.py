#!/usr/bin/env python3
"""Filter labels.csv to drop segments with bogus video timing.

Reads the fallback_per_segment.csv produced by
build_perperson_video_features.py and identifies labels.csv entries
whose underlying video frame didn't exist (12/12 frames in either slot
fell back to CLS — the signature of out-of-range second values).

Also produces a problematic_files.csv summarizing which (subject_id,
session, video_path) groups have bogus timing, so the recording-team
deep-dive has a starting point.

Usage:
    python scripts/filter_labels_by_fallback.py \\
        --labels data/labels.csv \\
        --fallback data/perperson_video_features_conf005/fallback_per_segment.csv \\
        --output-labels data/labels_filtered.csv \\
        --output-problems data/problematic_recordings.csv \\
        --max-fallback-frames 11

`--max-fallback-frames 11` means: drop segments with 12/12 fallback in
either slot (i.e., strictly out-of-range frames). Set to a smaller value
to also drop partially-broken segments — e.g. 6 drops segments where
either slot is more than half fallback.
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, required=True,
                    help="Input labels.csv")
    ap.add_argument("--fallback", type=Path, required=True,
                    help="fallback_per_segment.csv from build_perperson_video_features.py")
    ap.add_argument("--output-labels", type=Path, required=True,
                    help="Filtered labels.csv to write")
    ap.add_argument("--output-problems", type=Path, default=None,
                    help="Optional: per-(subject, session, video_path) summary of bogus rates")
    ap.add_argument(
        "--mode", choices=["segment", "subject"], default="segment",
        help="segment (default): drop individual bogus segments. subject: "
             "drop ALL segments from any (subject, session) whose bogus rate "
             "exceeds --max-subject-fallback-rate. Subject mode preserves "
             "time/content distribution within remaining subjects at the "
             "cost of dropping more data per fully-broken dyad — but avoids "
             "the early-session content bias that segment mode introduces "
             "for partially-broken dyads."
    )
    ap.add_argument("--max-fallback-frames", type=int, default=11,
                    help="[segment mode] Drop segments where either slot has > this "
                         "many fallback frames (out of 12). Default 11 = drop only "
                         "12/12 (strictly out-of-range frames).")
    ap.add_argument("--max-subject-fallback-rate", type=float, default=0.80,
                    help="[subject mode] Drop (subject, session) recordings whose "
                         "mean fallback rate (across both slots) exceeds this. "
                         "Default 0.80 = drop the ~7 fully-broken dyads.")
    args = ap.parse_args()

    # Build the (video_path, second) → fallback counts mapping from the
    # diagnostic. Each segment has two rows (slot 0, slot 1); we keep both.
    bogus_segments = set()  # (video_path, int(second)) where either slot exceeds threshold
    seg_diag = defaultdict(lambda: [0, 0])  # (vp, s) -> [slot0, slot1]
    # Per-(video_path, subject_id, session) totals for computing bogus rates.
    # Each segment contributes two slot rows; we count fallback frames separately
    # then normalize.
    rec_diag = defaultdict(lambda: {"n_total_frames": 0, "n_fallback_frames": 0})
    with open(args.fallback) as f:
        for r in csv.DictReader(f):
            vp = r["video_path"]
            sec = int(float(r["second"]))
            slot = int(r["slot"])
            n_fb = int(r["n_fallback_frames"])
            sub = r.get("subject_id", "")
            sess = r.get("session", "")
            seg_diag[(vp, sec)][slot] = n_fb
            if n_fb > args.max_fallback_frames:
                bogus_segments.add((vp, sec))
            rec_key = (vp, sub, sess)
            rec_diag[rec_key]["n_total_frames"] += int(r["n_total_frames"])
            rec_diag[rec_key]["n_fallback_frames"] += n_fb
    logger.info(
        f"Fallback diagnostic: {len(seg_diag):,} unique segments, "
        f"{len(rec_diag):,} unique recordings."
    )

    # Compute per-recording bogus rate and (in subject mode) the set of
    # recordings to drop entirely.
    bogus_recordings = set()
    if args.mode == "subject":
        for rec_key, d in rec_diag.items():
            rate = d["n_fallback_frames"] / d["n_total_frames"] if d["n_total_frames"] > 0 else 0
            if rate > args.max_subject_fallback_rate:
                bogus_recordings.add(rec_key)
        logger.info(
            f"Subject mode: dropping {len(bogus_recordings)} recordings with "
            f"fallback rate > {args.max_subject_fallback_rate*100:.0f}%."
        )
    else:
        logger.info(
            f"Segment mode: dropping {len(bogus_segments):,} segments with "
            f">{args.max_fallback_frames} fallback frames in either slot."
        )

    # Filter labels.csv.
    n_in = 0
    n_kept = 0
    bogus_subjects = defaultdict(lambda: {"n_total": 0, "n_bogus": 0, "max_bogus_second": 0})
    with open(args.labels) as fin, open(args.output_labels, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for r in reader:
            n_in += 1
            seg_key = (r["video_path"], int(r["second"]))
            sub = r.get("subject_id", "")
            sess = r.get("session", "")
            rec_key = (r["video_path"], sub, sess)
            grp = (sub, sess, r["video_path"])
            bogus_subjects[grp]["n_total"] += 1

            drop = False
            if args.mode == "subject":
                if rec_key in bogus_recordings:
                    drop = True
            else:
                if seg_key in bogus_segments:
                    drop = True

            if drop:
                bogus_subjects[grp]["n_bogus"] += 1
                bogus_subjects[grp]["max_bogus_second"] = max(
                    bogus_subjects[grp]["max_bogus_second"], int(r["second"])
                )
                continue
            writer.writerow(r)
            n_kept += 1
    n_dropped = n_in - n_kept
    logger.info(f"labels.csv: {n_in:,} rows in, {n_kept:,} kept, {n_dropped:,} dropped "
                f"({n_dropped/n_in*100:.1f}%).")
    logger.info(f"Wrote: {args.output_labels}")

    # Per-(subject, session, video_path) problem report.
    if args.output_problems is not None:
        rows = []
        for (sub, sess, vp), d in bogus_subjects.items():
            if d["n_total"] == 0:
                continue
            rate = d["n_bogus"] / d["n_total"]
            rows.append({
                "subject_id": sub,
                "session": sess,
                "video_path": vp,
                "n_total_segments": d["n_total"],
                "n_bogus_segments": d["n_bogus"],
                "bogus_rate": round(rate, 4),
                "max_bogus_second": d["max_bogus_second"],
            })
        # Sort worst-first.
        rows.sort(key=lambda r: -r["bogus_rate"])
        with open(args.output_problems, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Wrote per-recording problem report: {args.output_problems}")
        broken = [r for r in rows if r["bogus_rate"] > 0.95]
        partial = [r for r in rows if 0.10 < r["bogus_rate"] <= 0.95]
        clean = [r for r in rows if r["bogus_rate"] <= 0.10]
        print()
        print(f"Recording-level summary:")
        print(f"  Fully broken (>95% bogus):  {len(broken):>3d} recordings")
        print(f"  Partially broken (10-95%):  {len(partial):>3d} recordings")
        print(f"  Clean (<10% bogus):         {len(clean):>3d} recordings")


if __name__ == "__main__":
    main()

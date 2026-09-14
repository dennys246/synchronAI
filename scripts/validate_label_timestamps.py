#!/usr/bin/env python3
"""Check that every `second` in a labels CSV falls inside its video.

Reads container duration via ffprobe (metadata only, no frame decoding) and
compares it against the maximum labelled second for each video. Use it as a
guard before feature extraction: extracting against out-of-range seconds
silently yields degenerate features rather than an error.

Two labels files can be compared in one pass (--compare-labels) to show what a
repair changed.

Example:
    python scripts/validate_label_timestamps.py \
        --labels-file data/labels_care_repaired.csv \
        --compare-labels data/labels.csv \
        --path-from /storage1/fs1/perlmansusan/Active/moochie \
        --path-to /Volumes/perlmansusan/Active/moochie \
        --output docs/results/label_timestamp_validation.csv
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def max_second_per_video(labels_file):
    """Map video_path -> (max second, row count, subject_id)."""
    out = {}
    with open(labels_file) as fh:
        for row in csv.DictReader(fh):
            try:
                sec = int(float(row["second"]))
            except (KeyError, TypeError, ValueError):
                continue
            path = row["video_path"]
            prev = out.get(path)
            if prev is None:
                out[path] = [sec, 1, row.get("subject_id", "")]
            else:
                prev[0] = max(prev[0], sec)
                prev[1] += 1
    return out


def probe_duration(path):
    """Container duration in seconds, or None if ffprobe cannot read it."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--compare-labels", help="Second labels file to show alongside")
    parser.add_argument("--path-from", default="", help="Path prefix to rewrite")
    parser.add_argument("--path-to", default="", help="Replacement prefix")
    parser.add_argument("--output", help="Write per-video results to this CSV")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Seconds a label may exceed duration before it counts as overrun")
    args = parser.parse_args()

    primary = max_second_per_video(args.labels_file)
    compare = max_second_per_video(args.compare_labels) if args.compare_labels else {}

    results = []
    for video_path, (max_sec, n_rows, subject) in sorted(primary.items()):
        local = video_path
        if args.path_from:
            local = local.replace(args.path_from, args.path_to, 1)
        duration = probe_duration(local) if Path(local).exists() else None
        overrun = (max_sec - duration) if duration else None
        results.append({
            "subject_id": subject,
            "video_path": video_path,
            "n_rows": n_rows,
            "duration_s": round(duration, 1) if duration else "",
            "max_second": max_sec,
            "compare_max_second": compare.get(video_path, ["", "", ""])[0],
            "overrun_s": round(overrun, 1) if overrun is not None else "",
            "ratio": round(max_sec / duration, 2) if duration else "",
            "status": (
                "NO_VIDEO" if duration is None
                else "OVERRUN" if overrun > args.tolerance
                else "ok"
            ),
        })

    counts = defaultdict(int)
    rows_by_status = defaultdict(int)
    for r in results:
        counts[r["status"]] += 1
        rows_by_status[r["status"]] += r["n_rows"]

    total_rows = sum(r["n_rows"] for r in results)
    print(f"labels file : {args.labels_file}")
    print(f"videos      : {len(results)}")
    print(f"rows        : {total_rows:,}\n")
    for status in ("ok", "OVERRUN", "NO_VIDEO"):
        if counts[status]:
            share = rows_by_status[status] / total_rows if total_rows else 0
            print(f"  {status:9s} {counts[status]:4d} videos  "
                  f"{rows_by_status[status]:7,} rows ({share:.1%})")

    worst = sorted(
        (r for r in results if r["status"] == "OVERRUN"),
        key=lambda r: r["ratio"], reverse=True,
    )[:10]
    if worst:
        print("\nworst overruns:")
        for r in worst:
            print(f"  {r['subject_id']:8s} max_second {r['max_second']:>7} "
                  f"vs duration {r['duration_s']:>8}s  ({r['ratio']}x)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {args.output}")

    return 1 if counts["OVERRUN"] else 0


if __name__ == "__main__":
    sys.exit(main())

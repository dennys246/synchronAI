#!/usr/bin/env python3
"""Average source-video file size for the videos used in audio training.

Reads an audio feature index (the same CSV the multi-modal trainer joins
against), dedupes the unique source video files, splits them by study
(CARE vs P-CAT/R56), and reports count / total / average file size per study.

Note: as of writing, every wavlm audio feature set contains only CARE videos
(49 files, 42 families) and zero P-CAT/R56 videos. This script will surface
that directly, and will pick up P-CAT automatically if those features are
ever extracted.

Usage:
    python scripts/avg_video_size.py
    python scripts/avg_video_size.py --feature-dir data/wavlm_large_features
    python scripts/avg_video_size.py --study P-CAT
"""
import argparse
import csv
from pathlib import Path

# CSV video_path values use the cluster mount; on the Mac the same share is
# mounted at /Volumes/... . Translate so file stats work from either machine.
CLUSTER_PREFIX = "/storage1/fs1/perlmansusan/Active/moochie"
MAC_PREFIX = "/Volumes/perlmansusan/Active/moochie"


def resolve(path: str) -> Path:
    """Map a cluster path to whichever mount actually exists locally."""
    p = Path(path)
    if p.exists():
        return p
    if path.startswith(CLUSTER_PREFIX):
        alt = Path(MAC_PREFIX + path[len(CLUSTER_PREFIX):])
        if alt.exists():
            return alt
    elif path.startswith(MAC_PREFIX):
        alt = Path(CLUSTER_PREFIX + path[len(MAC_PREFIX):])
        if alt.exists():
            return alt
    return p  # return original; caller handles missing


def study_of(path: str) -> str:
    if "/P-CAT/" in path or "/R56/" in path:
        return "P-CAT"
    if "/CARE/" in path:
        return "CARE"
    return "other"


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-dir", default="data/wavlm_baseplus_features",
                    help="Audio feature dir holding feature_index.csv "
                         "(default: data/wavlm_baseplus_features)")
    ap.add_argument("--study", choices=["CARE", "P-CAT", "all"], default="all",
                    help="Restrict report to one study (default: all)")
    args = ap.parse_args()

    index = Path(args.feature_dir) / "feature_index.csv"
    if not index.exists():
        raise SystemExit(f"ERROR: {index} not found")

    # Dedupe source video paths from the feature index.
    videos: set[str] = set()
    with index.open(newline="") as f:
        reader = csv.DictReader(f)
        if "video_path" not in reader.fieldnames:
            raise SystemExit(f"ERROR: {index} has no 'video_path' column "
                             f"(columns: {reader.fieldnames})")
        for row in reader:
            videos.add(row["video_path"])

    # Group unique videos by study, stat each file once.
    by_study: dict[str, list[tuple[str, int]]] = {}
    missing: list[str] = []
    for vp in sorted(videos):
        st = study_of(vp)
        if args.study != "all" and st != args.study:
            continue
        local = resolve(vp)
        if not local.exists():
            missing.append(vp)
            continue
        by_study.setdefault(st, []).append((vp, local.stat().st_size))

    print(f"Feature index : {index}")
    print(f"Unique source videos in index: {len(videos)}")
    if args.study != "all":
        print(f"Filter        : study == {args.study}")
    print()

    if not by_study:
        print("No videos matched (and none could be stat'd).")
        if missing:
            print(f"{len(missing)} video file(s) could not be found on disk.")
        return

    grand_n = grand_bytes = 0
    for st in sorted(by_study):
        sizes = [s for _, s in by_study[st]]
        total = sum(sizes)
        n = len(sizes)
        grand_n += n
        grand_bytes += total
        print(f"[{st}]")
        print(f"  videos        : {n}")
        print(f"  total size    : {human(total)}")
        print(f"  average size  : {human(total / n)}")
        print(f"  min / max     : {human(min(sizes))} / {human(max(sizes))}")
        print()

    if len([s for s in by_study]) > 1:
        print("[ALL studies combined]")
        print(f"  videos        : {grand_n}")
        print(f"  total size    : {human(grand_bytes)}")
        print(f"  average size  : {human(grand_bytes / grand_n)}")
        print()

    if missing:
        print(f"WARNING: {len(missing)} video file(s) not found on this mount "
              f"(skipped). First few:")
        for vp in missing[:5]:
            print(f"  {vp}")


if __name__ == "__main__":
    main()

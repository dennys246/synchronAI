#!/usr/bin/env python3
"""
Build per-second turn-structure features from P-CAT R01 human verbal VAD coding.

Stage 1a of the audio language branch. No ASR, no diarization model, no LLM —
the speaker attribution is human-coded ground truth, so this is a pure
transform of an existing annotation file.

Input is the Datavyu-style long export (`verbal_*.csv`, git-ignored):

    record_id,timepoint,task,block,coder,column_name,onset,offset,content

`ParentVerbal` / `ChildVerbal` are two parallel tiers that fully tile each
trial; `content` is y/n (speaking / not). onset and offset are MILLISECONDS on
the same wall-clock as the video and as the synchrony coding's decoded Time
column. `Trial` rows delimit the coded region.

Output is a packed feature dir (`features_packed.bin` + `features_meta.json` +
`feature_index.csv` with a `row_idx` column), matching the fNIRS convention in
`src/synchronai/data/fnirs/feature_dataset.py`. Packed rather than one .pt per
second because 149 records x ~32 min of coded audio is ~280k entries, and this
repo already suffers on GPFS with 59k-file dirs.

Per entry the tensor is (2*window+1, 9): one row per second in a window centred
on the labelled second, and 9 binary channels:

    0 parent_speaking      4 parent_turn_onset     8 covered
    1 child_speaking       5 child_turn_onset
    2 both_speaking        6 parent_turn_offset
    3 neither_speaking     7 child_turn_offset

Channel 8 is load-bearing: it distinguishes "coded, nobody spoke" from "no
coding here". Without it a window overhanging the coded region would be
indistinguishable from silence — the zero-fill-means-missing trap that has
already cost this project two weeks once.

Usage:
    python scripts/extract_turn_features.py \
        --verbal-csv verbal_pcat_r01_07-27-2026.csv \
        --output-dir data/turntaking_features_r01
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOCAL_ROOT = "/Volumes/perlmansusan/Active/moochie"
CLUSTER_ROOT = "/storage1/fs1/perlmansusan/Active/moochie"
VIDEO_ROOT = f"{LOCAL_ROOT}/study_data/P-CAT/R01/data/WUSTL_data/T1/video_data/dbdos"

N_CHANNELS = 9
PARENT, CHILD = "ParentVerbal", "ChildVerbal"


def resolve_video(record_id: str, video_root: Path) -> str | None:
    """Find the DB-DOS recording for a record, tolerating the known anomalies.

    Rejects AppleDouble sidecars, partial-transfer artifacts (a hash suffix
    *after* the extension), zero-byte remux failures, and the loudness-fixed
    derivatives (heterogeneous audio, incl. 8 kHz mono). Accepts the 'ddbos'
    typo. Returns the longest candidate when a session was split across files,
    which is the part the coding timeline refers to.
    """
    d = video_root / record_id
    if not d.is_dir():
        return None
    cands = []
    for p in d.iterdir():
        n = p.name
        if n.startswith("._") or "_fixed" in n:
            continue
        if not re.search(r"\.(mp4|mkv)$", n, re.IGNORECASE):
            continue  # also drops *.mkv.02BF8B48 artifacts
        if not re.search(r"d[db]dos", n, re.IGNORECASE):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue
        cands.append((size, p))
    if not cands:
        return None
    return str(max(cands)[1])


def to_cluster(path: str) -> str:
    return path.replace(LOCAL_ROOT, CLUSTER_ROOT, 1)


def load_verbal(csv_path: Path) -> tuple[dict, dict]:
    """Return (per-record speech intervals, per-record coded spans) in seconds.

    Drops rows the audit flagged as unusable: offset <= onset (2 rows with
    offset written as 0), content '999' (39 missing/uncodable rows), and empty
    content (1 row). Counts them so the drop is visible, never silent.
    """
    speech: dict = defaultdict(lambda: {PARENT: [], CHILD: []})
    trials: dict = defaultdict(list)
    dropped = defaultdict(int)
    n = 0

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            n += 1
            rid, col, content = row["record_id"].strip(), row["column_name"], row["content"].strip()
            try:
                on, off = float(row["onset"]), float(row["offset"])
            except ValueError:
                dropped["unparseable_time"] += 1
                continue
            if off <= on:
                dropped["offset_le_onset"] += 1
                continue
            if col == "Trial":
                trials[rid].append((on / 1000.0, off / 1000.0))
                continue
            if col not in (PARENT, CHILD):
                dropped["unknown_column"] += 1
                continue
            if content == "999":
                dropped["content_999"] += 1
                continue
            if content == "":
                dropped["content_empty"] += 1
                continue
            if content == "y":
                speech[rid][col].append((on / 1000.0, off / 1000.0))
            elif content != "n":
                dropped["content_other"] += 1

    logger.info("Read %d rows; dropped %s", n, dict(dropped) or "nothing")
    return speech, trials


def timelines(intervals: list, lo: int, hi: int) -> np.ndarray:
    """Binary per-second activity over [lo, hi). A second counts as active if
    any speech interval overlaps it."""
    out = np.zeros(hi - lo, dtype=bool)
    for a, b in intervals:
        i0, i1 = max(lo, int(np.floor(a))), min(hi, int(np.ceil(b)))
        if i1 > i0:
            out[i0 - lo:i1 - lo] = True
    return out


def build_record(p_act: np.ndarray, c_act: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Stack the 9 per-second channels for a whole record."""
    ch = np.zeros((len(p_act), N_CHANNELS), dtype=np.float32)
    p = p_act & covered
    c = c_act & covered
    ch[:, 0] = p
    ch[:, 1] = c
    ch[:, 2] = p & c
    ch[:, 3] = covered & ~p & ~c
    prev_p = np.concatenate(([False], p[:-1]))
    prev_c = np.concatenate(([False], c[:-1]))
    ch[:, 4] = p & ~prev_p
    ch[:, 5] = c & ~prev_c
    ch[:, 6] = ~p & prev_p
    ch[:, 7] = ~c & prev_c
    ch[:, 8] = covered
    return ch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbal-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--video-root", default=VIDEO_ROOT)
    ap.add_argument("--window", type=int, default=5,
                    help="Half-width in seconds; entry is (2*window+1, 9). Default 5.")
    ap.add_argument("--require-video", action="store_true",
                    help="Skip records with no resolvable recording (default: keep, "
                         "with an empty video_path, so coding-only records stay usable).")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)
    W = 2 * args.window + 1

    speech, trials = load_verbal(Path(args.verbal_csv))
    records = sorted(set(speech) | set(trials))
    logger.info("Records with verbal coding: %d", len(records))

    per_record, index_rows, no_video = {}, [], []
    for rid in records:
        if not trials[rid]:
            logger.warning("%s: no Trial rows, skipping (no authoritative coded span)", rid)
            continue
        hi = int(np.ceil(max(b for _, b in trials[rid])))
        covered = timelines(trials[rid], 0, hi)
        if not covered.any():
            continue
        ch = build_record(
            timelines(speech[rid][PARENT], 0, hi),
            timelines(speech[rid][CHILD], 0, hi),
            covered,
        )
        vpath = resolve_video(rid, video_root)
        if vpath is None:
            no_video.append(rid)
            if args.require_video:
                continue
        per_record[rid] = ch
        for s in np.where(covered)[0]:
            index_rows.append({
                "record_id": rid,
                "video_path": to_cluster(vpath) if vpath else "",
                "second": int(s),
                "feature_dim": N_CHANNELS,
                "n_frames": W,
            })

    if not index_rows:
        logger.error("No entries produced.")
        return 1

    n = len(index_rows)
    logger.info("Packing %d entries of shape (%d, %d)...", n, W, N_CHANNELS)
    packed = np.memmap(
        out / "features_packed.bin", mode="w+", dtype=np.float32, shape=(n, W, N_CHANNELS),
    )
    for i, row in enumerate(index_rows):
        ch = per_record[row["record_id"]]
        s, T = row["second"], len(ch)
        win = np.zeros((W, N_CHANNELS), dtype=np.float32)
        lo, hi = s - args.window, s + args.window + 1
        src0, src1 = max(0, lo), min(T, hi)
        if src1 > src0:
            win[src0 - lo: src1 - lo] = ch[src0:src1]
        packed[i] = win
        row["row_idx"] = i
    packed.flush()
    del packed

    with open(out / "features_meta.json", "w") as f:
        json.dump({"shape": [n, W, N_CHANNELS], "dtype": "float32",
                   "channels": ["parent_speaking", "child_speaking", "both_speaking",
                                "neither_speaking", "parent_turn_onset", "child_turn_onset",
                                "parent_turn_offset", "child_turn_offset", "covered"],
                   "window_half_width_sec": args.window}, f, indent=2)

    cols = ["record_id", "video_path", "second", "feature_dim", "n_frames", "row_idx"]
    with open(out / "feature_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(index_rows)

    mb = n * W * N_CHANNELS * 4 / 1e6
    logger.info("Wrote %d entries (%.1f MB) to %s", n, mb, out)
    logger.info("Records packed: %d; without a resolvable recording: %d %s",
                len(per_record), len(no_video), no_video[:10] if no_video else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())

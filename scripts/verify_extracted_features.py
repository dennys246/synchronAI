#!/usr/bin/env python3
"""
Check an extracted feature dir for silent corruption before trusting it.

Extraction "succeeding" means the job exited 0. It does not mean the features
are real. `read_window_frames_dinov2` catches a failed frame read, logs a
warning, and substitutes a BLACK FRAME (processing.py, the `except` around
`get_frame_at_timestamp`); `extract_audio_features` zero-pads past the end of a
recording. Both produce a full-size, correctly-shaped, entirely wrong tensor,
and this repo has already lost weeks to a feature pipeline that zero-filled on
"thing not found".

Reports, per subject:
  * dead      - entries whose frames are all identical (a constant/black window)
  * near-zero - entries whose overall magnitude is ~0
  * frame-std - spread across frames within a window; ~0 means a frozen image
Optionally compares the same statistics against a reference dir (e.g. the CARE
features) so "normal" is measured rather than assumed.

Usage:
    python scripts/verify_extracted_features.py data/dinov2_features_meanpatch_r01pcat
    python scripts/verify_extracted_features.py <dir> --reference data/dinov2_features_meanpatch
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np
import torch


def scan(feature_dir: Path, limit: int | None):
    idx = list(csv.DictReader(open(feature_dir / "feature_index.csv")))
    if limit:
        idx = idx[:limit]
    per_sub = collections.defaultdict(lambda: dict(n=0, dead=0, nearzero=0, fstd=[], mag=[]))
    for r in idx:
        p = feature_dir / "features" / r["feature_file"]
        if not p.exists():
            continue
        x = torch.load(p, map_location="cpu", weights_only=True).detach().float().numpy()
        s = per_sub[r.get("subject_id", "?")]
        s["n"] += 1
        mag = float(np.abs(x).mean())
        s["mag"].append(mag)
        if mag < 1e-6:
            s["nearzero"] += 1
        if x.ndim >= 2 and x.shape[0] > 1:
            fs = float(x.reshape(x.shape[0], -1).std(axis=0).mean())
            s["fstd"].append(fs)
            if fs < 1e-6:
                s["dead"] += 1
    return per_sub


def report(name: str, per_sub) -> int:
    print(f"\n=== {name} ===")
    print(f"{'subject':14s} {'n':>6s} {'dead':>6s} {'near0':>6s} {'frame-std':>10s} {'|feat|':>9s}")
    bad = 0
    for sub in sorted(per_sub):
        s = per_sub[sub]
        fstd = np.mean(s["fstd"]) if s["fstd"] else float("nan")
        mag = np.mean(s["mag"]) if s["mag"] else float("nan")
        flag = ""
        if s["dead"] or s["nearzero"]:
            flag = "   <-- SUSPECT"
            bad += 1
        print(f"{sub:14s} {s['n']:6d} {s['dead']:6d} {s['nearzero']:6d} "
              f"{fstd:10.5f} {mag:9.4f}{flag}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feature_dir")
    ap.add_argument("--reference", help="Known-good dir to compare distributions against.")
    ap.add_argument("--limit", type=int, default=0, help="Only scan the first N entries.")
    args = ap.parse_args()

    d = Path(args.feature_dir)
    per_sub = scan(d, args.limit or None)
    if not per_sub:
        raise SystemExit(f"ERROR: no readable features under {d}")
    bad = report(d.name, per_sub)

    if args.reference:
        ref = scan(Path(args.reference), args.limit or 200)
        report(Path(args.reference).name + "  (reference)", ref)
        a = np.mean([v for s in per_sub.values() for v in s["fstd"]] or [np.nan])
        b = np.mean([v for s in ref.values() for v in s["fstd"]] or [np.nan])
        print(f"\nmean frame-std  target={a:.5f}  reference={b:.5f}  ratio={a / b:.2f}")
        if a < 0.25 * b:
            print("  WARNING: target frames vary far less than the reference — "
                  "likely frozen or black frames.")

    if bad:
        print(f"\n{bad} subject(s) flagged. Do NOT train on this dir until explained.")
        return 1
    print("\nNo dead or near-zero entries found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

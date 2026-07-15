#!/usr/bin/env python3
"""Pre-extract preprocessed per-pair fNIRS windows to a packed memmap cache.

The generative diffusion trainer re-runs HRfunc preprocessing on every raw
recording every epoch — the dominant cost (~90 min/epoch on the full dataset).
This builds that work ONCE: discover -> QC -> preprocess -> tile into raw
(target_len, 2) per-pair windows -> stream to a single packed binary, plus a
subject-indexed CSV. The trainer then memmaps it and each epoch is just
train (a few min), not preprocess.

Stores RAW (unnormalized) windows so the trainer computes its own running stats
and normalizes per batch — identical normalization behavior to the live path.

Output layout (mirrors features_packed.bin so a np.memmap reader works):
    <out>/windows_packed.bin   float32, shape (N, target_len, 2)
    <out>/windows_meta.json    {shape, dtype, per_entry_shape, sfreq_hz, ...}
    <out>/window_index.csv     row_idx, fnirs_path, subject_id, study,
                               participant_type, window_idx

Usage (cluster):
    $SYNCHRONAI_DIR/ml-env/bin/python scripts/cache_fnirs_windows.py \
        --output-dir $SYNCHRONAI_DIR/data/fnirs_perpair_window_cache \
        --qc-cache   $SYNCHRONAI_DIR/data/qc_tiers.csv \
        --sci-threshold 0.40 --snr-threshold 2.0 --no-require-cardiac
    # add --limit 20 for a quick dry-run build.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from synchronai.data.fnirs.dataset import load_training_windows  # noqa: E402
from synchronai.data.metadata import (  # noqa: E402
    MODALITY_FNIRS,
    classify_recording,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# load_training_windows logs per recording; quiet it during the bulk build.
logging.getLogger("synchronai.data.fnirs.dataset").setLevel(logging.WARNING)
logging.getLogger("synchronai.data.fnirs.quality_control").setLevel(logging.WARNING)

# Same 8 study roots the generative training / dataset summary use (cluster paths).
DEFAULT_STUDY_ROOTS = [
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dirs", default=None, help="Colon-separated roots; default = the 8 cluster study dirs.")
    p.add_argument("--qc-cache", default=None, help="Path to qc_tiers.csv (skips inline QC for cached recordings).")
    p.add_argument("--duration-seconds", type=float, default=60.0)
    p.add_argument("--target-sfreq", type=float, default=None)
    p.add_argument("--sci-threshold", type=float, default=0.40)
    p.add_argument("--snr-threshold", type=float, default=2.0)
    p.add_argument("--cardiac-peak-ratio", type=float, default=2.0)
    p.add_argument("--no-require-cardiac", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Process at most N recordings (0 = all). For dry-run.")
    args = p.parse_args()

    # Discover via the same logic the extractor uses.
    from extract_fnirs_features import discover_fnirs_paths  # noqa: E402 (script-local import)

    roots = args.data_dirs or ":".join(DEFAULT_STUDY_ROOTS)
    paths = discover_fnirs_paths(roots)
    if args.limit:
        paths = paths[: args.limit]
    logger.info("Discovered %d recordings to cache", len(paths))
    if not paths:
        logger.error("No recordings discovered. Check --data-dirs.")
        return 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / "windows_packed.bin"

    qc_kwargs = dict(
        enable_qc=True,
        sci_threshold=args.sci_threshold,
        snr_threshold=args.snr_threshold,
        cardiac_peak_ratio=args.cardiac_peak_ratio,
        require_cardiac=not args.no_require_cardiac,
        qc_cache_path=args.qc_cache,
    )

    per_entry_shape = None
    row_idx = 0
    n_recordings_used = 0
    n_skipped = 0
    index_rows = []

    with open(bin_path, "wb") as bf:
        for path in tqdm(paths, desc="Caching windows"):
            try:
                data = load_training_windows(
                    [path],
                    duration_seconds=args.duration_seconds,
                    target_sfreq_hz=args.target_sfreq,
                    per_pair=True,
                    normalize=False,
                    windowing="tile",
                    phase_offset=0,
                    **qc_kwargs,
                )
            except Exception as e:
                logger.warning("Skipping %s: %s", path, e)
                n_skipped += 1
                continue

            w = data.windows  # (M, target_len, 2) raw per-pair windows
            if w.size == 0:
                n_skipped += 1
                continue

            shape = tuple(w.shape[1:])
            if per_entry_shape is None:
                per_entry_shape = shape
            elif shape != per_entry_shape:
                logger.error(
                    "Window shape mismatch for %s: %s != %s (inconsistent sfreq/duration?) — skipping",
                    path, shape, per_entry_shape,
                )
                n_skipped += 1
                continue

            meta = classify_recording(path, modality=MODALITY_FNIRS).to_dict()
            subj = meta.get("subject_id", "")
            study = meta.get("study", "")
            ptype = meta.get("participant_type", "")

            w = np.ascontiguousarray(w, dtype=np.float32)
            bf.write(w.tobytes())
            for wi in range(w.shape[0]):
                index_rows.append((row_idx, path, subj, study, ptype, wi))
                row_idx += 1
            n_recordings_used += 1

    if per_entry_shape is None:
        logger.error("No valid windows produced — nothing cached.")
        return 1

    total_shape = [row_idx, int(per_entry_shape[0]), int(per_entry_shape[1])]
    with open(out / "windows_meta.json", "w") as f:
        json.dump({
            "shape": total_shape,
            "dtype": "float32",
            "per_entry_shape": [int(per_entry_shape[0]), int(per_entry_shape[1])],
            "sfreq_hz": args.target_sfreq or 7.8125,
            "duration_seconds": args.duration_seconds,
            "normalized": False,
            "n_recordings": n_recordings_used,
        }, f, indent=2)

    with open(out / "window_index.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["row_idx", "fnirs_path", "subject_id", "study", "participant_type", "window_idx"])
        wcsv.writerows(index_rows)

    size_gb = bin_path.stat().st_size / 1e9
    logger.info(
        "Cached %d windows from %d recordings (%d skipped) -> %s (%.2f GB), per_entry_shape=%s",
        row_idx, n_recordings_used, n_skipped, bin_path, size_gb, per_entry_shape,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

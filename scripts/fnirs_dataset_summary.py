#!/usr/bin/env python3
"""
fNIRS dataset summary statistics.

Walks the configured fNIRS study roots, reads recording durations from MNE
headers (no HRfunc preprocessing), optionally joins with the QC tier cache,
and prints a pre-QC + post-QC summary broken down by study, timepoint, and
task.

Uses the same discovery logic as scripts/extract_fnirs_features.py, so the
set of recordings counted here matches exactly what extraction would see.

Usage:
    python scripts/fnirs_dataset_summary.py
    python scripts/fnirs_dataset_summary.py --output runs/dataset_summary.csv
    python scripts/fnirs_dataset_summary.py --qc-cache data/qc_tiers.csv
    python scripts/fnirs_dataset_summary.py --data-dirs "dir1:dir2"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Ensure src/ is importable when run directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from synchronai.data.metadata import (  # noqa: E402
    MODALITY_FNIRS,
    classify_recording as _classify_recording_shared,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Default study roots — matches the 8 directories extract_fnirs_features.py processes.
# Uses cluster paths; override with --data-dirs when running elsewhere.
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


def discover_fnirs_paths(data_dirs: str) -> list[str]:
    """Walk directories and discover fNIRS recordings.

    Mirrors the logic in scripts/extract_fnirs_features.py.discover_fnirs_paths:
    NIRx directories (contain .hdr) take priority; .snirf/.fif files inside
    a NIRx dir are skipped to avoid double-counting.
    """
    paths: list[str] = []
    nirx_dirs: set[str] = set()

    for dir_path in data_dirs.split(":"):
        dir_path = dir_path.strip()
        if not dir_path or not Path(dir_path).exists():
            logger.warning("Skipping missing directory: %s", dir_path)
            continue

        # os.walk instead of Path.walk() for Python 3.11 compatibility
        for root, _dirs, files in sorted(os.walk(dir_path)):
            root_path = Path(root)
            if any(f.endswith(".hdr") for f in files):
                paths.append(str(root_path))
                nirx_dirs.add(str(root_path))
                continue
            if str(root_path) in nirx_dirs:
                continue
            for fname in sorted(files):
                if fname.endswith(".snirf") or fname.endswith(".fif"):
                    # Exclude deconvolved outputs (hemodynamic-only summary)
                    if "_Deconvolved" in fname or "deconvolved" in fname.lower():
                        continue
                    paths.append(str(root_path / fname))

    return paths


def classify_recording(path: str) -> dict:
    """Thin wrapper around the shared classifier returning a dict.

    The dict-form preserves the legacy API used by this script's downstream
    code. `path` is retained as an alias for `source_path` for back-compat.
    """
    meta = _classify_recording_shared(path, modality=MODALITY_FNIRS)
    d = meta.to_dict()
    d["path"] = d.get("source_path", path)
    return d


def get_duration_minutes(path: str) -> Optional[float]:
    """Return recording duration in minutes via an MNE header read.

    preload=False avoids loading the sample data — we only need n_times and
    sfreq from the header, so this should take a second or two per recording.
    Returns None if the file can't be read.
    """
    try:
        import mne
        p = Path(path)
        if p.is_dir():
            raw = mne.io.read_raw_nirx(str(p), preload=False, verbose="ERROR")
        elif str(p).lower().endswith(".snirf"):
            raw = mne.io.read_raw_snirf(str(p), preload=False, verbose="ERROR")
        elif str(p).lower().endswith(".fif"):
            raw = mne.io.read_raw_fif(str(p), preload=False, verbose="ERROR")
        else:
            return None
        sfreq = float(raw.info["sfreq"])
        if sfreq <= 0:
            return None
        seconds = raw.n_times / sfreq
        return seconds / 60.0
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def load_qc_cache(path: str) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["fnirs_path"].astype(str), df["quality_tier"].astype(str)))


def _per_study_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study, sub in df.groupby("study", sort=False):
        rows.append({
            "study": study,
            "recordings": len(sub),
            "subjects_total": sub["subject_id"].nunique(),
            "children": sub[sub["participant_type"] == "child"]["subject_id"].nunique(),
            "adults": sub[sub["participant_type"] == "adult"]["subject_id"].nunique(),
            "dyads": sub[sub["participant_type"] == "dyad"]["subject_id"].nunique(),
            "families": sub["family_id"].replace("", pd.NA).dropna().nunique(),
            "total_hours": round(sub["duration_minutes"].sum() / 60.0, 1),
            "mean_min": round(sub["duration_minutes"].mean(), 1) if len(sub) else 0.0,
        })
    total = {
        "study": "TOTAL",
        "recordings": len(df),
        "subjects_total": df["subject_id"].nunique(),
        "children": df[df["participant_type"] == "child"]["subject_id"].nunique(),
        "adults": df[df["participant_type"] == "adult"]["subject_id"].nunique(),
        "dyads": df[df["participant_type"] == "dyad"]["subject_id"].nunique(),
        "families": df["family_id"].replace("", pd.NA).dropna().nunique(),
        "total_hours": round(df["duration_minutes"].sum() / 60.0, 1),
        "mean_min": round(df["duration_minutes"].mean(), 1) if len(df) else 0.0,
    }
    return pd.DataFrame(rows + [total])


def _per_study_timepoint_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, tp, task), sub in df.groupby(["study", "timepoint", "task"], sort=False):
        rows.append({
            "study": study,
            "timepoint": tp,
            "task": task,
            "recordings": len(sub),
            "subjects": sub["subject_id"].nunique(),
            "children": sub[sub["participant_type"] == "child"]["subject_id"].nunique(),
            "adults": sub[sub["participant_type"] == "adult"]["subject_id"].nunique(),
            "total_hours": round(sub["duration_minutes"].sum() / 60.0, 1),
        })
    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame, label: str) -> None:
    print()
    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)

    if len(df) == 0:
        print("(no recordings)")
        return

    total_min = df["duration_minutes"].sum()
    print(f"Total recordings:        {len(df)}")
    print(f"Unique subjects (stable across visits): {df['subject_id'].nunique()}")
    print(f"  Children:              {df[df['participant_type'] == 'child']['subject_id'].nunique()}")
    print(f"  Adults:                {df[df['participant_type'] == 'adult']['subject_id'].nunique()}")
    print(f"  Dyads (family-level):  {df[df['participant_type'] == 'dyad']['subject_id'].nunique()}")
    print(f"Unique families:         {df['family_id'].replace('', pd.NA).dropna().nunique()}")
    print(f"Total duration:          {total_min:.1f} min  ({total_min / 60:.1f} hours)")
    print(f"Per-recording duration:  min={df['duration_minutes'].min():.1f}, "
          f"mean={df['duration_minutes'].mean():.1f}, "
          f"median={df['duration_minutes'].median():.1f}, "
          f"max={df['duration_minutes'].max():.1f} min")

    print("\nPer-study:")
    print(_per_study_rows(df).to_string(index=False))

    print("\nPer-study × timepoint × task:")
    print(_per_study_timepoint_rows(df).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dirs", default=None,
        help="Colon-separated study roots. Defaults to the cluster paths used by the extraction scripts.",
    )
    parser.add_argument(
        "--qc-cache", default=None,
        help="Path to qc_tiers.csv for the post-QC breakdown. Defaults to data/qc_tiers.csv if it exists.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional per-recording CSV output (study, timepoint, duration, tier, ...).",
    )
    args = parser.parse_args()

    roots = args.data_dirs.split(":") if args.data_dirs else list(DEFAULT_STUDY_ROOTS)
    logger.info("Scanning %d study roots", len(roots))

    # QC cache
    qc_cache_path = args.qc_cache
    if qc_cache_path is None and Path("data/qc_tiers.csv").exists():
        qc_cache_path = "data/qc_tiers.csv"

    qc_map: dict[str, str] = {}
    if qc_cache_path and Path(qc_cache_path).exists():
        logger.info("Loading QC cache from %s", qc_cache_path)
        qc_map = load_qc_cache(qc_cache_path)
        logger.info("Loaded %d QC tier entries", len(qc_map))
    else:
        logger.warning(
            "No QC cache found — post-QC breakdown will be skipped "
            "(pass --qc-cache path/to/qc_tiers.csv to enable it)"
        )

    # Discover
    logger.info("Discovering fNIRS recordings...")
    paths = discover_fnirs_paths(":".join(roots))
    logger.info("Discovered %d recordings", len(paths))
    if not paths:
        logger.error("No recordings found. Check --data-dirs.")
        return 1

    # Classify + read durations
    records = []
    for path in tqdm(paths, desc="Reading headers"):
        meta = classify_recording(path)
        meta["duration_minutes"] = get_duration_minutes(path) or 0.0
        meta["quality_tier"] = qc_map.get(path, "unknown")
        records.append(meta)

    df = pd.DataFrame(records)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info("Wrote per-recording data to %s", args.output)

    # Pre-QC summary (all discovered)
    _print_summary(df, "Pre-QC (all discovered recordings)")

    # Post-QC summary (exclude rejected)
    if qc_map:
        df_postqc = df[df["quality_tier"].isin(["gold", "standard", "salvageable"])].copy()
        _print_summary(df_postqc, "Post-QC (gold / standard / salvageable)")

        # Also show tier distribution
        print()
        print("Tier distribution:")
        tier_counts = df["quality_tier"].value_counts().to_dict()
        for tier in ("gold", "standard", "salvageable", "rejected", "unknown"):
            c = tier_counts.get(tier, 0)
            if c:
                pct = 100 * c / len(df)
                print(f"  {tier:12s} {c:5d}  ({pct:.1f}%)")

    # Sanity checks
    n_unknown_study = (df["study"] == "unknown").sum()
    n_unknown_participant = (df["participant_type"] == "unknown").sum()
    if n_unknown_study or n_unknown_participant:
        print()
        print("Sanity checks:")
        if n_unknown_study:
            print(f"  WARNING: {n_unknown_study} recordings with unclassified study — inspect paths.")
        if n_unknown_participant:
            print(f"  WARNING: {n_unknown_participant} recordings with unknown participant_type.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

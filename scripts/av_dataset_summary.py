#!/usr/bin/env python3
"""Video/audio dataset summary statistics.

Reads `data/av_manifest.csv` (produced by `scripts/build_av_manifest.py`) and
prints a breakdown by study / modality / timepoint / task with subject,
family, and duration counts.

This is the video/audio analogue of `scripts/fnirs_dataset_summary.py`.

Usage:
    python scripts/av_dataset_summary.py
    python scripts/av_dataset_summary.py --manifest data/av_manifest.csv
    python scripts/av_dataset_summary.py --manifest data/av_manifest.csv \\
        --output runs/av_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _duration_minutes(df: pd.DataFrame) -> pd.Series:
    """Return duration in minutes; falls back to 0 for rows without probe data."""
    if "duration_seconds" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0.0) / 60.0


def _per_study_modality_rows(df: pd.DataFrame) -> pd.DataFrame:
    dur_min = _duration_minutes(df)
    rows = []
    for (study, modality), sub in df.groupby(["study", "modality"], sort=False):
        sub_dur = dur_min.loc[sub.index]
        rows.append({
            "study": study,
            "modality": modality,
            "files": len(sub),
            "subjects": sub["subject_id"].replace("", pd.NA).dropna().nunique(),
            "families": sub["family_id"].replace("", pd.NA).dropna().nunique(),
            "children": sub[sub["participant_type"] == "child"]["subject_id"].nunique(),
            "adults": sub[sub["participant_type"] == "adult"]["subject_id"].nunique(),
            "dyads": sub[sub["participant_type"] == "dyad"]["subject_id"].nunique(),
            "total_hours": round(sub_dur.sum() / 60.0, 1),
            "mean_min": round(sub_dur.mean(), 1) if len(sub) else 0.0,
        })
    total_dur = dur_min
    rows.append({
        "study": "TOTAL",
        "modality": "all",
        "files": len(df),
        "subjects": df["subject_id"].replace("", pd.NA).dropna().nunique(),
        "families": df["family_id"].replace("", pd.NA).dropna().nunique(),
        "children": df[df["participant_type"] == "child"]["subject_id"].nunique(),
        "adults": df[df["participant_type"] == "adult"]["subject_id"].nunique(),
        "dyads": df[df["participant_type"] == "dyad"]["subject_id"].nunique(),
        "total_hours": round(total_dur.sum() / 60.0, 1),
        "mean_min": round(total_dur.mean(), 1) if len(df) else 0.0,
    })
    return pd.DataFrame(rows)


def _per_study_timepoint_task_rows(df: pd.DataFrame) -> pd.DataFrame:
    dur_min = _duration_minutes(df)
    rows = []
    grouped = df.groupby(["study", "modality", "timepoint", "task"], sort=False)
    for (study, modality, tp, task), sub in grouped:
        sub_dur = dur_min.loc[sub.index]
        rows.append({
            "study": study,
            "modality": modality,
            "timepoint": tp,
            "task": task,
            "files": len(sub),
            "subjects": sub["subject_id"].replace("", pd.NA).dropna().nunique(),
            "families": sub["family_id"].replace("", pd.NA).dropna().nunique(),
            "total_hours": round(sub_dur.sum() / 60.0, 1),
        })
    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame) -> None:
    if len(df) == 0:
        print("(manifest is empty)")
        return

    # Separate participant recordings from calibration/demo samples so the
    # main numbers reflect real dataset size.
    samples = df[df["participant_type"] == "sample"]
    participants = df[df["participant_type"] != "sample"]

    p_dur = _duration_minutes(participants)
    total_min = p_dur.sum()

    print()
    print("=" * 78)
    print("  Video / Audio Dataset Summary")
    print("=" * 78)
    print(f"Total files:             {len(participants)}  (excludes {len(samples)} calibration/demo samples)")
    print(f"  video:                 {(participants['modality'] == 'video').sum()}")
    print(f"  audio:                 {(participants['modality'] == 'audio').sum()}")
    print(f"Unique subjects:         {participants['subject_id'].replace('', pd.NA).dropna().nunique()}")
    print(f"  children:              {participants[participants['participant_type'] == 'child']['subject_id'].nunique()}")
    print(f"  adults:                {participants[participants['participant_type'] == 'adult']['subject_id'].nunique()}")
    print(f"  dyads (family-level):  {participants[participants['participant_type'] == 'dyad']['subject_id'].nunique()}")
    print(f"Unique families:         {participants['family_id'].replace('', pd.NA).dropna().nunique()}")
    if total_min > 0:
        print(f"Total duration:          {total_min:.1f} min  ({total_min / 60:.1f} hours)")
        print(f"Per-file duration:       min={p_dur[p_dur > 0].min():.1f}, "
              f"mean={p_dur[p_dur > 0].mean():.1f}, "
              f"median={p_dur[p_dur > 0].median():.1f}, "
              f"max={p_dur.max():.1f} min "
              f"(computed over {(p_dur > 0).sum()} probed files)")
    else:
        print("Total duration:          (not available — run build_av_manifest.py without --no-probe)")

    print("\nPer-study × modality (participant recordings only):")
    print(_per_study_modality_rows(participants).to_string(index=False))

    print("\nPer-study × modality × timepoint × task (participant recordings only):")
    print(_per_study_timepoint_task_rows(participants).to_string(index=False))

    if len(samples):
        s_dur = _duration_minutes(samples)
        print(f"\nCalibration / demo samples (not participant data): {len(samples)} files, "
              f"{s_dur.sum():.1f} min")
        s_breakdown = samples.groupby(["study", "modality", "task"], sort=False).size().reset_index(name="files")
        print(s_breakdown.to_string(index=False))

    # Sanity checks
    n_unknown_study = int((df["study"] == "unknown").sum())
    n_unknown_pt = int((df["participant_type"] == "unknown").sum())
    n_unknown_task = int((df["task"] == "unknown").sum())
    if n_unknown_study or n_unknown_pt or n_unknown_task:
        print()
        print("Sanity checks:")
        if n_unknown_study:
            print(f"  WARNING: {n_unknown_study} files with unclassified study.")
        if n_unknown_pt:
            print(f"  WARNING: {n_unknown_pt} files with unknown participant_type.")
        if n_unknown_task:
            print(f"  WARNING: {n_unknown_task} files with unknown task.")

        # Show a sample of unresolved files so the classifier can be extended.
        mask = (
            (df["study"] == "unknown")
            | (df["participant_type"] == "unknown")
            | (df["task"] == "unknown")
        )
        sample = df.loc[mask, ["study", "modality", "timepoint", "task", "source_path"]].head(25)
        if len(sample):
            print()
            print(f"Sample of unresolved files (up to 25 of {int(mask.sum())}):")
            for _, row in sample.iterrows():
                flags = []
                if row["study"] == "unknown":
                    flags.append("study")
                if row["task"] == "unknown":
                    flags.append("task")
                print(f"  [{','.join(flags) or 'pt':10s}] {row['source_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/av_manifest.csv",
        help="Path to av_manifest.csv produced by build_av_manifest.py.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: write the per-study × timepoint × task breakdown to CSV.",
    )
    parser.add_argument(
        "--dump-unresolved", default=None,
        help="Optional path: write every file with unknown study/task/participant_type "
             "to this CSV so the classifier can be extended to cover them.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(
            "Manifest not found: %s\nRun `python scripts/build_av_manifest.py` first.",
            manifest_path,
        )
        return 1

    # Force identifier columns to string so pandas doesn't coerce e.g.
    # family_id="1144" into 1144.0 when some rows are empty.
    df = pd.read_csv(
        manifest_path,
        dtype={
            "family_id": str,
            "subject_id": str,
            "recording_id": str,
            "session_id": str,
            "timepoint": str,
            "task": str,
            "study": str,
            "site": str,
            "modality": str,
            "participant_type": str,
            "part": str,
        },
        keep_default_na=False,
    )
    logger.info("Loaded %d manifest rows", len(df))

    _print_summary(df)

    if args.output:
        out = _per_study_timepoint_task_rows(df)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        logger.info("Wrote per-study breakdown to %s", args.output)

    if args.dump_unresolved:
        mask = (
            (df["study"] == "unknown")
            | (df["participant_type"] == "unknown")
            | (df["task"] == "unknown")
        )
        unresolved = df.loc[mask].copy()
        Path(args.dump_unresolved).parent.mkdir(parents=True, exist_ok=True)
        unresolved.to_csv(args.dump_unresolved, index=False)
        logger.info("Wrote %d unresolved rows to %s", len(unresolved), args.dump_unresolved)

    return 0


if __name__ == "__main__":
    sys.exit(main())

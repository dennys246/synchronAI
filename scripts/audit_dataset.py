#!/usr/bin/env python3
"""Comprehensive dataset audit for the synchronAI project.

Analyses the labels CSV used for binary dyadic synchrony prediction,
producing statistics, data-quality checks, split simulations, and
visualisations.  All heavy computation is pandas-based; plots use
matplotlib with the Agg backend so the script runs headless.

Usage:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --labels-file scripts/data/labels.csv --output-dir runs/audit/
    python scripts/audit_dataset.py --check-videos   # also verify video files on disk
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {"video_path", "second", "label", "subject_id", "session"}
LABEL_NAMES = {0: "async", 1: "sync"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_serialisable(obj: Any) -> Any:
    """Recursively convert numpy / pandas types so json.dumps works."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serialisable(v) for v in obj]
    return obj


def _pct(count: int, total: int) -> float:
    """Return percentage rounded to two decimals, safe for zero total."""
    if total == 0:
        return 0.0
    return round(100.0 * count / total, 2)


# ---------------------------------------------------------------------------
# Overall statistics
# ---------------------------------------------------------------------------

def compute_overall_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return top-level dataset statistics."""
    total = len(df)
    n_subjects = df["subject_id"].nunique()
    n_videos = df["video_path"].nunique()
    n_sessions = df["session"].nunique()

    class_counts = df["label"].value_counts().sort_index().to_dict()
    class_pcts = {
        lab: _pct(cnt, total)
        for lab, cnt in class_counts.items()
    }
    minority = min(class_counts.values())
    majority = max(class_counts.values())
    balance_ratio = round(minority / majority, 4) if majority > 0 else 0.0

    return {
        "total_labeled_seconds": total,
        "n_unique_subjects": n_subjects,
        "n_unique_videos": n_videos,
        "n_unique_sessions": n_sessions,
        "class_counts": class_counts,
        "class_percentages": class_pcts,
        "class_balance_ratio": balance_ratio,
        "label_names": {int(k): LABEL_NAMES.get(k, f"class_{k}") for k in class_counts},
    }


# ---------------------------------------------------------------------------
# Per-subject analysis
# ---------------------------------------------------------------------------

def compute_per_subject_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return a per-subject DataFrame and an aggregate summary dict."""
    grouped = df.groupby("subject_id")

    per_subj = pd.DataFrame({
        "n_samples": grouped.size(),
        "n_sync": grouped["label"].sum(),
    })
    per_subj["n_async"] = per_subj["n_samples"] - per_subj["n_sync"]
    per_subj["sync_rate"] = (per_subj["n_sync"] / per_subj["n_samples"]).round(4)
    per_subj["n_videos"] = grouped["video_path"].nunique()
    per_subj = per_subj.reset_index()

    samples = per_subj["n_samples"]
    sync_rates = per_subj["sync_rate"]

    extreme_high = per_subj.loc[sync_rates > 0.90, "subject_id"].tolist()
    extreme_low = per_subj.loc[sync_rates < 0.10, "subject_id"].tolist()
    few_samples_threshold = 10
    few_samples = per_subj.loc[samples < few_samples_threshold, "subject_id"].tolist()

    summary = {
        "samples_per_subject": {
            "min": int(samples.min()),
            "max": int(samples.max()),
            "mean": round(float(samples.mean()), 2),
            "median": float(samples.median()),
            "std": round(float(samples.std()), 2),
        },
        "sync_rate_per_subject": {
            "min": round(float(sync_rates.min()), 4),
            "max": round(float(sync_rates.max()), 4),
            "mean": round(float(sync_rates.mean()), 4),
            "median": round(float(sync_rates.median()), 4),
            "std": round(float(sync_rates.std()), 4),
        },
        "subjects_with_extreme_sync_rate_gt90pct": extreme_high,
        "subjects_with_extreme_sync_rate_lt10pct": extreme_low,
        "subjects_with_fewer_than_10_samples": few_samples,
        "n_subjects_extreme_high": len(extreme_high),
        "n_subjects_extreme_low": len(extreme_low),
        "n_subjects_few_samples": len(few_samples),
    }
    return per_subj, summary


# ---------------------------------------------------------------------------
# Per-video analysis
# ---------------------------------------------------------------------------

def compute_per_video_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return a per-video DataFrame and an aggregate summary dict."""
    grouped = df.groupby("video_path")

    per_vid = pd.DataFrame({
        "n_labeled_seconds": grouped.size(),
        "min_second": grouped["second"].min(),
        "max_second": grouped["second"].max(),
        "n_sync": grouped["label"].sum(),
        "subject_id": grouped["subject_id"].first(),
        "session": grouped["session"].first(),
    })
    per_vid["n_async"] = per_vid["n_labeled_seconds"] - per_vid["n_sync"]
    per_vid["sync_rate"] = (per_vid["n_sync"] / per_vid["n_labeled_seconds"]).round(4)
    per_vid["span_seconds"] = per_vid["max_second"] - per_vid["min_second"] + 1
    per_vid["coverage"] = (per_vid["n_labeled_seconds"] / per_vid["span_seconds"]).round(4)

    # Temporal gaps per video — largest gap between consecutive labelled seconds
    def _max_gap(g: pd.DataFrame) -> int:
        secs = g["second"].sort_values().values
        if len(secs) < 2:
            return 0
        return int(np.max(np.diff(secs)))

    max_gaps = grouped.apply(_max_gap).rename("max_gap")
    per_vid = per_vid.join(max_gaps)

    # Mono-label videos
    per_vid["mono_label"] = (per_vid["n_sync"] == 0) | (per_vid["n_async"] == 0)
    per_vid = per_vid.reset_index()

    labeled_secs = per_vid["n_labeled_seconds"]
    summary = {
        "labeled_seconds_per_video": {
            "min": int(labeled_secs.min()),
            "max": int(labeled_secs.max()),
            "mean": round(float(labeled_secs.mean()), 2),
            "median": float(labeled_secs.median()),
        },
        "coverage": {
            "min": round(float(per_vid["coverage"].min()), 4),
            "max": round(float(per_vid["coverage"].max()), 4),
            "mean": round(float(per_vid["coverage"].mean()), 4),
        },
        "max_gap": {
            "min": int(per_vid["max_gap"].min()),
            "max": int(per_vid["max_gap"].max()),
            "mean": round(float(per_vid["max_gap"].mean()), 2),
        },
        "n_mono_label_videos": int(per_vid["mono_label"].sum()),
        "mono_label_videos": per_vid.loc[per_vid["mono_label"], "video_path"].tolist(),
    }
    return per_vid, summary


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

def compute_temporal_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse temporal properties of the labelled seconds."""
    seconds = df["second"].values

    # Basic distribution
    second_stats = {
        "min": int(seconds.min()),
        "max": int(seconds.max()),
        "mean": round(float(seconds.mean()), 2),
        "median": float(np.median(seconds)),
        "std": round(float(seconds.std()), 2),
    }

    # Temporal autocorrelation within each video
    autocorr_values: List[float] = []
    for _, grp in df.sort_values(["video_path", "second"]).groupby("video_path"):
        labels = grp["label"].values
        if len(labels) < 2:
            continue
        # Agreement between adjacent labels
        matches = (labels[:-1] == labels[1:]).astype(float)
        autocorr_values.append(float(matches.mean()))

    autocorr_summary = {}
    if autocorr_values:
        arr = np.array(autocorr_values)
        autocorr_summary = {
            "mean_adjacent_agreement": round(float(arr.mean()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "std": round(float(arr.std()), 4),
        }

    # Run-length analysis (consecutive same-label runs within each video)
    run_lengths_sync: List[int] = []
    run_lengths_async: List[int] = []
    for _, grp in df.sort_values(["video_path", "second"]).groupby("video_path"):
        labels = grp["label"].values
        if len(labels) == 0:
            continue
        current_label = labels[0]
        current_run = 1
        for lab in labels[1:]:
            if lab == current_label:
                current_run += 1
            else:
                if current_label == 1:
                    run_lengths_sync.append(current_run)
                else:
                    run_lengths_async.append(current_run)
                current_label = lab
                current_run = 1
        # final run
        if current_label == 1:
            run_lengths_sync.append(current_run)
        else:
            run_lengths_async.append(current_run)

    def _run_summary(runs: List[int]) -> Dict[str, Any]:
        if not runs:
            return {"count": 0}
        arr = np.array(runs)
        return {
            "count": len(runs),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": round(float(arr.mean()), 2),
            "median": float(np.median(arr)),
            "std": round(float(arr.std()), 2),
        }

    return {
        "second_distribution": second_stats,
        "temporal_autocorrelation": autocorr_summary,
        "run_lengths_sync": _run_summary(run_lengths_sync),
        "run_lengths_async": _run_summary(run_lengths_async),
        # Store raw run lengths for plotting (not in JSON — handled separately)
        "_raw_run_lengths_sync": run_lengths_sync,
        "_raw_run_lengths_async": run_lengths_async,
    }


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Run a battery of data-quality checks and return findings."""
    issues: Dict[str, Any] = {}

    # 1. Duplicate entries
    dup_mask = df.duplicated(subset=["video_path", "second"], keep=False)
    n_dups = int(dup_mask.sum())
    issues["duplicate_entries"] = {
        "count": n_dups,
        "rows": df.loc[dup_mask].index.tolist()[:50],  # cap at 50 examples
    }

    # 2. Missing / null values
    nulls = df.isnull().sum().to_dict()
    issues["missing_values"] = {col: int(v) for col, v in nulls.items()}
    issues["total_missing"] = int(df.isnull().sum().sum())

    # 3. Invalid subject IDs
    invalid_sid = df.loc[df["subject_id"].isnull(), "subject_id"].index.tolist()
    issues["invalid_subject_ids"] = {
        "null_count": len(invalid_sid),
        "example_rows": invalid_sid[:20],
    }

    # 4. Sparse videos (<10 labelled seconds)
    vid_counts = df.groupby("video_path").size()
    sparse_videos = vid_counts[vid_counts < 10].index.tolist()
    issues["sparse_videos_lt10"] = {
        "count": len(sparse_videos),
        "videos": sparse_videos,
    }

    # 5. Potential label noise — isolated flips (s-a-s or a-s-a patterns)
    noise_count = 0
    noise_examples: List[Dict[str, Any]] = []
    for vpath, grp in df.sort_values(["video_path", "second"]).groupby("video_path"):
        labels = grp["label"].values
        indices = grp.index.values
        seconds = grp["second"].values
        if len(labels) < 3:
            continue
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
                noise_count += 1
                if len(noise_examples) < 30:
                    noise_examples.append({
                        "video_path": str(vpath),
                        "second": int(seconds[i]),
                        "label": int(labels[i]),
                        "surrounding_labels": [int(labels[i - 1]), int(labels[i]), int(labels[i + 1])],
                        "row_index": int(indices[i]),
                    })

    issues["potential_label_noise_isolated_flips"] = {
        "count": noise_count,
        "examples": noise_examples,
    }

    return issues


# ---------------------------------------------------------------------------
# Video file checks (optional)
# ---------------------------------------------------------------------------

def check_video_files(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify that referenced video files exist on disk."""
    unique_videos = df["video_path"].unique()
    missing: List[str] = []
    found: List[str] = []
    for vp in unique_videos:
        if Path(vp).is_file():
            found.append(vp)
        else:
            missing.append(vp)
    return {
        "total_unique_videos": len(unique_videos),
        "found_on_disk": len(found),
        "missing_on_disk": len(missing),
        "missing_video_paths": missing,
    }


# ---------------------------------------------------------------------------
# Split simulation
# ---------------------------------------------------------------------------

def simulate_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate a subject-based 80/20 split and report statistics."""
    rng = np.random.RandomState(seed)
    subjects = df["subject_id"].unique()
    rng.shuffle(subjects)

    n_train = int(len(subjects) * train_frac)
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:])

    train_df = df[df["subject_id"].isin(train_subjects)]
    val_df = df[df["subject_id"].isin(val_subjects)]

    def _split_stats(split_df: pd.DataFrame, split_subjects: set) -> Dict[str, Any]:
        total = len(split_df)
        class_counts = split_df["label"].value_counts().sort_index().to_dict()
        return {
            "n_subjects": len(split_subjects),
            "n_samples": total,
            "class_counts": class_counts,
            "class_percentages": {
                lab: _pct(cnt, total) for lab, cnt in class_counts.items()
            },
        }

    train_stats = _split_stats(train_df, train_subjects)
    val_stats = _split_stats(val_df, val_subjects)

    # Class imbalance difference
    train_sync_pct = train_stats["class_percentages"].get(1, 0.0)
    val_sync_pct = val_stats["class_percentages"].get(1, 0.0)
    imbalance_diff = round(abs(train_sync_pct - val_sync_pct), 2)

    return {
        "seed": seed,
        "train_fraction": train_frac,
        "train": train_stats,
        "val": val_stats,
        "sync_pct_diff_train_vs_val": imbalance_diff,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["label"].value_counts().sort_index()
    labels = [LABEL_NAMES.get(k, str(k)) for k in counts.index]
    colors = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors[:len(labels)], edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(df) * 0.005,
            f"{cnt:,}\n({_pct(cnt, len(df))}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    _save_fig(fig, out_dir / "class_distribution.png")


def plot_samples_per_subject(per_subj: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_subj["n_samples"], bins=40, color="#3498db", edgecolor="black", linewidth=0.5)
    ax.axvline(per_subj["n_samples"].median(), color="red", linestyle="--", label=f"median={per_subj['n_samples'].median():.0f}")
    ax.set_xlabel("Samples per Subject")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Distribution of Samples per Subject")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_dir / "samples_per_subject.png")


def plot_sync_rate_per_subject(per_subj: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_subj["sync_rate"], bins=40, color="#9b59b6", edgecolor="black", linewidth=0.5)
    ax.axvline(per_subj["sync_rate"].median(), color="red", linestyle="--", label=f"median={per_subj['sync_rate'].median():.2f}")
    ax.set_xlabel("Sync Rate (label=1 fraction)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Distribution of Per-Subject Sync Rate")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_dir / "sync_rate_per_subject.png")


def plot_temporal_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["second"], bins=60, color="#1abc9c", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Second (timestamp within video)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Labeled Second Values")
    fig.tight_layout()
    _save_fig(fig, out_dir / "temporal_distribution.png")


def plot_run_lengths(
    run_lengths_sync: List[int],
    run_lengths_async: List[int],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    if run_lengths_sync:
        axes[0].hist(run_lengths_sync, bins=range(1, max(run_lengths_sync) + 2),
                      color="#2ecc71", edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[0].set_title("Sync Run Lengths")
    axes[0].set_xlabel("Run Length (consecutive sync seconds)")
    axes[0].set_ylabel("Frequency")

    if run_lengths_async:
        axes[1].hist(run_lengths_async, bins=range(1, max(run_lengths_async) + 2),
                      color="#e74c3c", edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[1].set_title("Async Run Lengths")
    axes[1].set_xlabel("Run Length (consecutive async seconds)")
    axes[1].set_ylabel("Frequency")

    fig.suptitle("Distribution of Consecutive Run Lengths", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, out_dir / "run_lengths.png")


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_json_report(report: Dict[str, Any], path: Path) -> None:
    """Write the full audit report as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(_json_serialisable(report), f, indent=2, default=str)
    logger.info("Wrote JSON report: %s", path)


def write_summary_txt(report: Dict[str, Any], path: Path) -> None:
    """Write a human-readable summary text file."""
    lines: List[str] = []

    def _line(text: str = "") -> None:
        lines.append(text)

    _line("=" * 72)
    _line("synchronAI Dataset Audit Summary")
    _line("=" * 72)

    ov = report["overall"]
    _line()
    _line("OVERALL STATISTICS")
    _line("-" * 40)
    _line(f"  Total labeled seconds:   {ov['total_labeled_seconds']:,}")
    _line(f"  Unique subjects:         {ov['n_unique_subjects']}")
    _line(f"  Unique videos:           {ov['n_unique_videos']}")
    _line(f"  Unique sessions:         {ov['n_unique_sessions']}")
    _line(f"  Class balance ratio:     {ov['class_balance_ratio']}")
    _line()
    for lab, cnt in ov["class_counts"].items():
        name = ov["label_names"].get(str(lab), ov["label_names"].get(int(lab), str(lab)))
        pct = ov["class_percentages"][lab]
        _line(f"  Label {lab} ({name}): {cnt:>8,}  ({pct}%)")

    subj = report["per_subject_summary"]
    _line()
    _line("PER-SUBJECT ANALYSIS")
    _line("-" * 40)
    s = subj["samples_per_subject"]
    _line(f"  Samples/subject — min: {s['min']}, max: {s['max']}, "
          f"mean: {s['mean']}, median: {s['median']}, std: {s['std']}")
    sr = subj["sync_rate_per_subject"]
    _line(f"  Sync rate/subject — min: {sr['min']}, max: {sr['max']}, "
          f"mean: {sr['mean']}, median: {sr['median']}, std: {sr['std']}")
    _line(f"  Subjects with >90% sync rate:  {subj['n_subjects_extreme_high']}")
    _line(f"  Subjects with <10% sync rate:  {subj['n_subjects_extreme_low']}")
    _line(f"  Subjects with <10 samples:     {subj['n_subjects_few_samples']}")

    vid = report["per_video_summary"]
    _line()
    _line("PER-VIDEO ANALYSIS")
    _line("-" * 40)
    ls = vid["labeled_seconds_per_video"]
    _line(f"  Labeled secs/video — min: {ls['min']}, max: {ls['max']}, "
          f"mean: {ls['mean']}, median: {ls['median']}")
    cv = vid["coverage"]
    _line(f"  Temporal coverage — min: {cv['min']}, max: {cv['max']}, mean: {cv['mean']}")
    mg = vid["max_gap"]
    _line(f"  Max gap between labels — min: {mg['min']}, max: {mg['max']}, mean: {mg['mean']}")
    _line(f"  Mono-label videos:       {vid['n_mono_label_videos']}")

    temp = report["temporal"]
    _line()
    _line("TEMPORAL ANALYSIS")
    _line("-" * 40)
    sd = temp["second_distribution"]
    _line(f"  Second range: [{sd['min']}, {sd['max']}]  mean: {sd['mean']}  std: {sd['std']}")
    if temp["temporal_autocorrelation"]:
        ac = temp["temporal_autocorrelation"]
        _line(f"  Adjacent-label agreement: mean={ac['mean_adjacent_agreement']}, "
              f"std={ac['std']}")
    rl_s = temp["run_lengths_sync"]
    rl_a = temp["run_lengths_async"]
    _line(f"  Sync runs: {rl_s.get('count', 0)} runs, "
          f"mean length={rl_s.get('mean', 'N/A')}, max={rl_s.get('max', 'N/A')}")
    _line(f"  Async runs: {rl_a.get('count', 0)} runs, "
          f"mean length={rl_a.get('mean', 'N/A')}, max={rl_a.get('max', 'N/A')}")

    qc = report["quality_checks"]
    _line()
    _line("DATA QUALITY CHECKS")
    _line("-" * 40)
    _line(f"  Duplicate (video_path, second) entries: {qc['duplicate_entries']['count']}")
    _line(f"  Total missing/null values:              {qc['total_missing']}")
    for col, cnt in qc["missing_values"].items():
        if cnt > 0:
            _line(f"    — {col}: {cnt} missing")
    _line(f"  Null subject IDs:                       {qc['invalid_subject_ids']['null_count']}")
    _line(f"  Sparse videos (<10 labeled secs):       {qc['sparse_videos_lt10']['count']}")
    _line(f"  Potential label-noise isolated flips:    {qc['potential_label_noise_isolated_flips']['count']}")

    sp = report["split_simulation"]
    _line()
    _line("SPLIT SIMULATION (subject-based 80/20)")
    _line("-" * 40)
    _line(f"  Seed: {sp['seed']}")
    tr = sp["train"]
    _line(f"  Train — subjects: {tr['n_subjects']}, samples: {tr['n_samples']:,}")
    for lab, cnt in tr["class_counts"].items():
        _line(f"    Label {lab}: {cnt:>8,}  ({tr['class_percentages'][lab]}%)")
    va = sp["val"]
    _line(f"  Val   — subjects: {va['n_subjects']}, samples: {va['n_samples']:,}")
    for lab, cnt in va["class_counts"].items():
        _line(f"    Label {lab}: {cnt:>8,}  ({va['class_percentages'][lab]}%)")
    _line(f"  Sync % difference (train vs val): {sp['sync_pct_diff_train_vs_val']}pp")

    if "video_file_check" in report:
        vc = report["video_file_check"]
        _line()
        _line("VIDEO FILE CHECK")
        _line("-" * 40)
        _line(f"  Total unique videos: {vc['total_unique_videos']}")
        _line(f"  Found on disk:       {vc['found_on_disk']}")
        _line(f"  Missing on disk:     {vc['missing_on_disk']}")

    _line()
    _line("=" * 72)
    _line("End of Audit")
    _line("=" * 72)

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    logger.info("Wrote text summary: %s", path)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset audit for the synchronAI labels CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="scripts/data/labels.csv",
        help="Path to the labels CSV (default: scripts/data/labels.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/audit/",
        help="Directory for output files (default: runs/audit/)",
    )
    parser.add_argument(
        "--check-videos",
        action="store_true",
        default=False,
        help="Verify that video files exist on disk (disabled by default)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    labels_path = Path(args.labels_file)
    out_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not labels_path.is_file():
        logger.error("Labels file not found: %s", labels_path)
        sys.exit(1)

    logger.info("Loading labels from %s", labels_path)
    df = pd.read_csv(labels_path)

    # Validate columns
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        logger.error("Missing expected columns: %s", missing_cols)
        sys.exit(1)

    logger.info("Loaded %d rows with columns: %s", len(df), list(df.columns))

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir.resolve())

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    report: Dict[str, Any] = {}

    logger.info("Computing overall statistics ...")
    report["overall"] = compute_overall_stats(df)

    logger.info("Computing per-subject statistics ...")
    per_subj_df, per_subj_summary = compute_per_subject_stats(df)
    report["per_subject_summary"] = per_subj_summary

    logger.info("Computing per-video statistics ...")
    per_vid_df, per_vid_summary = compute_per_video_stats(df)
    report["per_video_summary"] = per_vid_summary

    logger.info("Computing temporal statistics ...")
    temporal = compute_temporal_stats(df)
    # Extract raw run lengths before storing in JSON report
    raw_sync_runs = temporal.pop("_raw_run_lengths_sync", [])
    raw_async_runs = temporal.pop("_raw_run_lengths_async", [])
    report["temporal"] = temporal

    logger.info("Running data quality checks ...")
    report["quality_checks"] = run_quality_checks(df)

    logger.info("Simulating subject-based 80/20 split ...")
    report["split_simulation"] = simulate_split(df)

    if args.check_videos:
        logger.info("Checking video files on disk ...")
        report["video_file_check"] = check_video_files(df)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    logger.info("Writing audit outputs ...")

    # JSON report
    write_json_report(report, out_dir / "audit_report.json")

    # Human-readable summary
    summary_text = write_summary_txt(report, out_dir / "audit_summary.txt")

    # Per-subject CSV
    per_subj_path = out_dir / "per_subject_stats.csv"
    per_subj_df.to_csv(per_subj_path, index=False)
    logger.info("Wrote per-subject CSV: %s", per_subj_path)

    # Per-video CSV
    per_vid_path = out_dir / "per_video_stats.csv"
    per_vid_df.to_csv(per_vid_path, index=False)
    logger.info("Wrote per-video CSV: %s", per_vid_path)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    logger.info("Generating visualisations ...")

    plot_class_distribution(df, out_dir)
    plot_samples_per_subject(per_subj_df, out_dir)
    plot_sync_rate_per_subject(per_subj_df, out_dir)
    plot_temporal_distribution(df, out_dir)
    plot_run_lengths(raw_sync_runs, raw_async_runs, out_dir)

    # ------------------------------------------------------------------
    # Print summary to stdout
    # ------------------------------------------------------------------
    print()
    print(summary_text)

    logger.info("Audit complete. All outputs saved to %s", out_dir.resolve())


if __name__ == "__main__":
    main()

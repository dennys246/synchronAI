"""
Data validation module for video classification datasets.

Validates:
- Labels CSV format and content
- Video file accessibility
- Frame extraction capability
- Label coverage and distribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from synchronai.data.video.processing import load_video_info

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_labels_csv(
    labels_file: Union[str, Path],
    required_columns: Optional[list[str]] = None,
) -> ValidationResult:
    """Validate labels CSV file format and content.

    Args:
        labels_file: Path to labels.csv
        required_columns: Required column names

    Returns:
        ValidationResult with errors and warnings
    """
    labels_file = Path(labels_file)
    errors = []
    warnings = []
    stats = {}

    if required_columns is None:
        required_columns = ["video_path", "second", "label"]

    # Check file exists
    if not labels_file.exists():
        return ValidationResult(
            valid=False,
            errors=[f"Labels file not found: {labels_file}"],
        )

    # Load CSV
    try:
        df = pd.read_csv(labels_file)
    except Exception as e:
        return ValidationResult(
            valid=False,
            errors=[f"Failed to read labels CSV: {e}"],
        )

    # Check required columns
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Basic stats
    stats["total_rows"] = len(df)
    stats["unique_videos"] = df["video_path"].nunique() if "video_path" in df.columns else 0
    stats["unique_subjects"] = df["subject_id"].nunique() if "subject_id" in df.columns else 0

    # Label distribution
    if "label" in df.columns:
        label_counts = df["label"].value_counts().to_dict()
        stats["label_distribution"] = label_counts

        # Check for class imbalance
        if len(label_counts) >= 2:
            counts = list(label_counts.values())
            ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
            stats["class_imbalance_ratio"] = ratio
            if ratio > 10:
                warnings.append(f"Severe class imbalance: {ratio:.1f}:1 ratio")
            elif ratio > 5:
                warnings.append(f"Moderate class imbalance: {ratio:.1f}:1 ratio")

    # Check for duplicates
    if "video_path" in df.columns and "second" in df.columns:
        duplicates = df.duplicated(subset=["video_path", "second"]).sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate (video_path, second) entries")

    # Check label values
    if "label" in df.columns:
        valid_labels = {0, 1}
        invalid_labels = set(df["label"].unique()) - valid_labels
        if invalid_labels:
            errors.append(f"Invalid label values: {invalid_labels} (expected 0 or 1)")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_video_files(
    labels_file: Union[str, Path],
    sample_videos: int = 5,
) -> ValidationResult:
    """Validate video files are accessible and readable.

    Args:
        labels_file: Path to labels.csv
        sample_videos: Number of videos to sample for validation

    Returns:
        ValidationResult with errors and warnings
    """
    labels_file = Path(labels_file)
    errors = []
    warnings = []
    stats = {}

    try:
        df = pd.read_csv(labels_file)
    except Exception as e:
        return ValidationResult(valid=False, errors=[f"Failed to read labels CSV: {e}"])

    if "video_path" not in df.columns:
        return ValidationResult(valid=False, errors=["video_path column not found"])

    unique_videos = df["video_path"].unique()
    stats["total_videos"] = len(unique_videos)

    # Check file existence
    missing_videos = []
    for video_path in unique_videos:
        if not Path(video_path).exists():
            missing_videos.append(video_path)

    if missing_videos:
        errors.append(f"{len(missing_videos)} video files not found")
        stats["missing_videos"] = missing_videos[:10]  # Show first 10

    # Sample video accessibility
    accessible_videos = [v for v in unique_videos if Path(v).exists()]
    sample_count = min(sample_videos, len(accessible_videos))

    if sample_count > 0:
        import random
        sampled = random.sample(list(accessible_videos), sample_count)

        readable_count = 0
        for video_path in sampled:
            try:
                info = load_video_info(video_path)
                if info.frame_count > 0 and info.fps > 0:
                    readable_count += 1
            except Exception as e:
                warnings.append(f"Failed to read {Path(video_path).name}: {e}")

        stats["sample_readable"] = f"{readable_count}/{sample_count}"
        if readable_count < sample_count:
            warnings.append(
                f"Only {readable_count}/{sample_count} sampled videos were readable"
            )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_dataset(
    labels_file: Union[str, Path],
    sample_videos: int = 5,
) -> ValidationResult:
    """Run full dataset validation.

    Args:
        labels_file: Path to labels.csv
        sample_videos: Number of videos to sample

    Returns:
        Combined ValidationResult
    """
    all_errors = []
    all_warnings = []
    all_stats = {}

    # Validate CSV format
    csv_result = validate_labels_csv(labels_file)
    all_errors.extend(csv_result.errors)
    all_warnings.extend(csv_result.warnings)
    all_stats["csv"] = csv_result.stats

    # Only validate videos if CSV is valid
    if csv_result.valid:
        video_result = validate_video_files(labels_file, sample_videos)
        all_errors.extend(video_result.errors)
        all_warnings.extend(video_result.warnings)
        all_stats["videos"] = video_result.stats

    return ValidationResult(
        valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        stats=all_stats,
    )


def print_validation_report(result: ValidationResult) -> None:
    """Print validation report to console.

    Args:
        result: ValidationResult to print
    """
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    if result.valid:
        print("\n✓ Dataset validation PASSED")
    else:
        print("\n✗ Dataset validation FAILED")

    if result.errors:
        print("\nERRORS:")
        for error in result.errors:
            print(f"  ✗ {error}")

    if result.warnings:
        print("\nWARNINGS:")
        for warning in result.warnings:
            print(f"  ! {warning}")

    if result.stats:
        print("\nSTATISTICS:")
        _print_stats(result.stats, indent=2)

    print("=" * 60 + "\n")


def _print_stats(stats: dict, indent: int = 0) -> None:
    """Recursively print stats dict."""
    prefix = " " * indent
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_stats(value, indent + 2)
        elif isinstance(value, list):
            print(f"{prefix}{key}: [{len(value)} items]")
            for item in value[:5]:
                print(f"{prefix}  - {item}")
            if len(value) > 5:
                print(f"{prefix}  ... and {len(value) - 5} more")
        else:
            print(f"{prefix}{key}: {value}")

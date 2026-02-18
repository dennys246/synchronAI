"""
Convert raw xlsx label files and video directory structure to labels.csv.

This module handles the specific data format used in the CARE synchrony study:
- Label directory: {subject_id}/{session}/{subject_id}_{session}_{activity}.xlsx
- Video directory: {subject_prefix}/{session}/{subject_id}_{session}_DB-DOS.mp4
- Subject ID mapping: 5-digit subject ID -> 4-digit video folder prefix
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RawDataConfig:
    """Configuration for raw data preprocessing."""

    label_dir: Path
    video_dir: Path
    output_csv: Path
    label_encoding: dict[str, int] = field(default_factory=lambda: {"a": 0, "s": 1})
    conflict_strategy: str = "last"  # last, first, or error
    video_filename_pattern: str = "{subject_id}_{session}_DB-DOS.mp4"
    min_labels_per_video: int = 10

    def __post_init__(self) -> None:
        self.label_dir = Path(self.label_dir)
        self.video_dir = Path(self.video_dir)
        self.output_csv = Path(self.output_csv)


@dataclass
class PreprocessingReport:
    """Report from preprocessing operation."""

    total_subjects: int = 0
    total_sessions: int = 0
    total_videos: int = 0
    total_labels: int = 0
    missing_videos: list[str] = field(default_factory=list)
    sparse_videos: list[str] = field(default_factory=list)
    label_distribution: dict[int, int] = field(default_factory=dict)
    labels_per_video: dict[str, int] = field(default_factory=dict)
    conflicts_found: int = 0


def subject_to_video_prefix(subject_id: str) -> str:
    """Map 5-digit subject ID to 4-digit video folder prefix.

    Example: '50021' -> '5002'
    """
    return subject_id[:4]


def load_label_xlsx(xlsx_path: Path, encoding: dict[str, int]) -> pd.DataFrame:
    """Load a single label xlsx file with flexible parsing.

    Tries multiple strategies to find timestamp and label columns:
    1. Simple two-column format (second, label) without headers
    2. Named columns matching common patterns
    3. Heuristic detection of numeric (time) and categorical (label) columns

    Args:
        xlsx_path: Path to xlsx file
        encoding: Mapping of label codes to integers (e.g., {'a': 0, 's': 1})

    Returns:
        DataFrame with columns ['second', 'label']

    Raises:
        ValueError: If no valid label data is found
    """
    # Skip macOS resource fork files
    if xlsx_path.name.startswith("._"):
        raise ValueError(f"Skipping macOS resource fork file: {xlsx_path}")

    label_codes = set(encoding.keys())

    # Try reading with headers first to inspect structure
    try:
        df_header = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    if df_header.empty:
        raise ValueError(f"Empty Excel file: {xlsx_path}")

    # Log file structure for debugging
    logger.debug(
        f"File {xlsx_path.name}: {len(df_header)} rows, "
        f"columns={list(df_header.columns)}"
    )

    # Strategy 1: Try headerless two-column format
    result = _try_simple_format(xlsx_path, encoding, label_codes)
    if result is not None and len(result) > 0:
        logger.debug(f"  -> Parsed as simple format: {len(result)} labels")
        return result

    # Strategy 2: Try finding columns by name
    result = _try_named_columns(df_header, encoding, label_codes)
    if result is not None and len(result) > 0:
        logger.debug(f"  -> Parsed by column names: {len(result)} labels")
        return result

    # Strategy 3: Heuristic - find any numeric + categorical column pair
    result = _try_heuristic_columns(df_header, encoding, label_codes)
    if result is not None and len(result) > 0:
        logger.debug(f"  -> Parsed by heuristic: {len(result)} labels")
        return result

    # Strategy 4: Try headerless read and scan all column pairs
    result = _try_headerless_scan(xlsx_path, encoding, label_codes)
    if result is not None and len(result) > 0:
        logger.debug(f"  -> Parsed headerless scan: {len(result)} labels")
        return result

    # Nothing worked - log detailed file structure for debugging
    logger.warning(
        f"Could not parse {xlsx_path.name}. "
        f"Columns: {list(df_header.columns)}"
    )
    # Show unique values per column (truncated)
    for col in df_header.columns:
        unique_vals = df_header[col].astype(str).str.lower().str.strip().unique()[:10]
        logger.warning(f"  Column '{col}': {list(unique_vals)}")

    raise ValueError(
        f"No valid label data found in {xlsx_path}. "
        f"Expected label codes: {label_codes}"
    )


def _try_simple_format(
    xlsx_path: Path, encoding: dict[str, int], label_codes: set[str]
) -> Optional[pd.DataFrame]:
    """Try parsing as simple two-column headerless format."""
    try:
        df = pd.read_excel(
            xlsx_path, header=None, names=["second", "label_code"], engine="openpyxl"
        )
    except Exception:
        return None

    if len(df.columns) < 2:
        return None

    # Check if first column is numeric (timestamps)
    df["second"] = pd.to_numeric(df["second"], errors="coerce")
    if df["second"].notna().mean() < 0.5:
        return None

    # Check if second column has valid labels
    df["label_code"] = df["label_code"].astype(str).str.lower().str.strip()
    df["label"] = df["label_code"].map(encoding)

    if df["label"].notna().sum() == 0:
        return None

    df = df.dropna(subset=["second", "label"])
    df["second"] = df["second"].astype(int)
    df["label"] = df["label"].astype(int)

    return df[["second", "label"]]


def _parse_time_to_seconds(time_series: pd.Series) -> pd.Series:
    """Convert time values to seconds.

    Handles:
    - Numeric values (already seconds)
    - HH:MM:SS or MM:SS time strings
    - datetime.time objects
    """
    result = pd.Series(index=time_series.index, dtype="Int64")

    for idx, val in time_series.items():
        if pd.isna(val):
            continue

        # Already numeric
        if isinstance(val, (int, float)):
            result[idx] = int(val)
            continue

        # Convert to string and parse
        val_str = str(val).strip()

        # Try HH:MM:SS or MM:SS format
        if ":" in val_str:
            parts = val_str.split(":")
            try:
                if len(parts) == 3:
                    h, m, s = int(parts[0]), int(parts[1]), int(float(parts[2]))
                    result[idx] = h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = int(parts[0]), int(float(parts[1]))
                    result[idx] = m * 60 + s
            except (ValueError, TypeError):
                continue
        else:
            # Try parsing as numeric
            try:
                result[idx] = int(float(val_str))
            except (ValueError, TypeError):
                continue

    return result


def _try_named_columns(
    df: pd.DataFrame, encoding: dict[str, int], label_codes: set[str]
) -> Optional[pd.DataFrame]:
    """Try finding columns by common name patterns."""
    # Common column names for timestamps
    time_names = {"second", "seconds", "time", "timestamp", "start", "frame", "sec"}
    # Common column names for labels
    label_names = {"label", "code", "sync", "synchrony", "rating", "score"}

    col_map = {str(c).lower().strip(): c for c in df.columns}

    # Find time column
    time_col = None
    for name in time_names:
        if name in col_map:
            time_col = col_map[name]
            break

    # Find label column by name first
    label_col = None
    for name in label_names:
        if name in col_map:
            label_col = col_map[name]
            break

    # If no label column by name, find one with valid label codes
    if label_col is None:
        for col in df.columns:
            if col == time_col:
                continue
            values = df[col].astype(str).str.lower().str.strip()
            if values.isin(label_codes).mean() > 0.1:  # Lower threshold
                label_col = col
                break

    if time_col is None or label_col is None:
        return None

    result = pd.DataFrame()
    # Parse time strings (HH:MM:SS) to seconds
    result["second"] = _parse_time_to_seconds(df[time_col])
    result["label_code"] = df[label_col].astype(str).str.lower().str.strip()
    result["label"] = result["label_code"].map(encoding)

    result = result.dropna(subset=["second", "label"])
    if len(result) == 0:
        return None

    result["second"] = result["second"].astype(int)
    result["label"] = result["label"].astype(int)

    return result[["second", "label"]]


def _try_heuristic_columns(
    df: pd.DataFrame, encoding: dict[str, int], label_codes: set[str]
) -> Optional[pd.DataFrame]:
    """Try finding columns by data characteristics."""
    # Find columns that look like timestamps (numeric or time strings)
    time_cols = []
    for col in df.columns:
        # Try parsing as time strings first
        parsed = _parse_time_to_seconds(df[col])
        valid_ratio = parsed.notna().mean()
        if valid_ratio > 0.5:
            time_cols.append((col, valid_ratio))

    # Find columns that contain label codes
    label_cols = []
    for col in df.columns:
        values = df[col].astype(str).str.lower().str.strip()
        match_ratio = values.isin(label_codes).mean()
        if match_ratio > 0.1:  # Lower threshold for heuristic
            label_cols.append((col, match_ratio))

    if not time_cols or not label_cols:
        return None

    # Use best time column and best matching label column
    time_col = max(time_cols, key=lambda x: x[1])[0]
    label_col = max(label_cols, key=lambda x: x[1])[0]

    result = pd.DataFrame()
    result["second"] = _parse_time_to_seconds(df[time_col])
    result["label_code"] = df[label_col].astype(str).str.lower().str.strip()
    result["label"] = result["label_code"].map(encoding)

    result = result.dropna(subset=["second", "label"])
    if len(result) == 0:
        return None

    result["second"] = result["second"].astype(int)
    result["label"] = result["label"].astype(int)

    return result[["second", "label"]]


def _try_headerless_scan(
    xlsx_path: Path, encoding: dict[str, int], label_codes: set[str]
) -> Optional[pd.DataFrame]:
    """Try headerless read and scan all adjacent column pairs."""
    try:
        df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")
    except Exception:
        return None

    if len(df.columns) < 2:
        return None

    best_result = None
    best_count = 0

    # Try each pair of adjacent columns
    for i in range(len(df.columns) - 1):
        col_time = df.columns[i]
        col_label = df.columns[i + 1]

        # Check if this pair works - use time parser for HH:MM:SS strings
        seconds = _parse_time_to_seconds(df[col_time])
        labels_str = df[col_label].astype(str).str.lower().str.strip()
        labels = labels_str.map(encoding)

        valid_mask = seconds.notna() & labels.notna()
        valid_count = valid_mask.sum()

        if valid_count > best_count:
            best_count = valid_count
            result = pd.DataFrame()
            result["second"] = seconds[valid_mask].astype(int)
            result["label"] = labels[valid_mask].astype(int)
            best_result = result.reset_index(drop=True)

    return best_result if best_count > 0 else None


def _filter_duration_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out duration marker rows that don't represent actual labels.

    Handles cases where a duration row exists (e.g., "00:20:00" = 1200s)
    among normal label timestamps (e.g., "00:15" = 15s).

    Args:
        df: DataFrame with 'second' and 'label' columns

    Returns:
        DataFrame with duration markers removed
    """
    if len(df) < 2:
        return df

    seconds = df["second"].values

    # Detect and filter outliers (likely duration markers)
    # If a value is > 10x the median and > 600 seconds (10 min), it's likely a duration
    median_val = np.median(seconds)
    if median_val > 0:
        outlier_mask = (seconds > 10 * median_val) & (seconds > 600)
        if outlier_mask.any():
            logger.debug(
                f"Filtering {outlier_mask.sum()} outlier timestamps "
                f"(likely duration markers)"
            )
            df = df[~outlier_mask].copy()

    return df


def combine_label_files(
    xlsx_paths: list[Path],
    encoding: dict[str, int],
    conflict_strategy: str = "last",
) -> tuple[pd.DataFrame, int]:
    """Combine multiple xlsx label files for a single video.

    Args:
        xlsx_paths: List of paths to xlsx files (sorted alphabetically)
        encoding: Label code mapping
        conflict_strategy: How to handle conflicting labels for same second
            - "last": Use label from last file (alphabetical)
            - "first": Use label from first file
            - "error": Raise error on conflict

    Returns:
        Tuple of (combined DataFrame with columns ['second', 'label'], conflict_count)

    Raises:
        ValueError: If conflict_strategy is "error" and conflicts are found
    """
    xlsx_paths = sorted(xlsx_paths)  # Alphabetical order
    combined = pd.DataFrame(columns=["second", "label"])
    conflict_count = 0

    for xlsx_path in xlsx_paths:
        try:
            df = load_label_xlsx(xlsx_path, encoding)
            df = _filter_duration_markers(df)  # Remove duration/metadata rows
        except Exception as e:
            logger.warning(f"Failed to load {xlsx_path}: {e}")
            continue

        if len(df) == 0:
            continue

        # Check for conflicts with existing data
        if len(combined) > 0:
            overlap = combined[combined["second"].isin(df["second"])]
            if len(overlap) > 0:
                merged = overlap.merge(df, on="second", suffixes=("_existing", "_new"))
                conflicts = merged[merged["label_existing"] != merged["label_new"]]

                if len(conflicts) > 0:
                    conflict_count += len(conflicts)
                    conflict_seconds = conflicts["second"].tolist()

                    if conflict_strategy == "error":
                        raise ValueError(
                            f"Conflicting labels at seconds {conflict_seconds} in {xlsx_path}"
                        )
                    else:
                        logger.debug(
                            f"Conflicting labels at seconds {conflict_seconds[:5]}... "
                            f"in {xlsx_path.name}, using {conflict_strategy}"
                        )

        if conflict_strategy == "last":
            # Remove existing entries for seconds in new file, then append
            combined = combined[~combined["second"].isin(df["second"])]
            combined = pd.concat([combined, df], ignore_index=True)
        else:  # first
            # Only add new seconds not already present
            new_seconds = df[~df["second"].isin(combined["second"])]
            combined = pd.concat([combined, new_seconds], ignore_index=True)

    return combined.sort_values("second").reset_index(drop=True), conflict_count


def discover_sessions(label_dir: Path) -> list[dict]:
    """Discover all subject/session combinations.

    Args:
        label_dir: Root label directory

    Returns:
        List of dicts with keys: subject_id, session, label_files
    """
    sessions = []

    if not label_dir.exists():
        logger.error(f"Label directory does not exist: {label_dir}")
        return sessions

    for subject_path in sorted(label_dir.iterdir()):
        if not subject_path.is_dir():
            continue
        # Skip hidden directories and non-subject folders
        if subject_path.name.startswith("."):
            continue

        subject_id = subject_path.name

        for session_path in sorted(subject_path.iterdir()):
            if not session_path.is_dir():
                continue
            if session_path.name.startswith("."):
                continue

            session = session_path.name

            # Find all xlsx files in this session
            xlsx_files = list(session_path.glob("*.xlsx"))
            # Filter out temporary Excel files and macOS resource forks
            xlsx_files = [
                f for f in xlsx_files
                if not f.name.startswith("~$") and not f.name.startswith("._")
            ]

            if xlsx_files:
                sessions.append(
                    {
                        "subject_id": subject_id,
                        "session": session,
                        "label_files": xlsx_files,
                    }
                )

    return sessions


def resolve_video_path(
    video_dir: Path,
    subject_id: str,
    session: str,
    filename_pattern: str,
) -> Optional[Path]:
    """Resolve video path from subject ID and session.

    Args:
        video_dir: Root video directory
        subject_id: Full subject ID (e.g., '50021')
        session: Session name (e.g., 'V0')
        filename_pattern: Video filename pattern with placeholders

    Returns:
        Path to video file, or None if not found
    """
    prefix = subject_to_video_prefix(subject_id)
    filename = filename_pattern.format(subject_id=subject_id, session=session)
    session_dir = video_dir / prefix / session

    # Check if session directory exists
    if not session_dir.exists():
        return None

    # Try exact match first
    video_path = session_dir / filename
    if video_path.exists():
        return video_path

    # Try alternative: glob for any video with subject_id and session
    alt_patterns = [
        f"{subject_id}_{session}_*.mp4",
        f"{subject_id}_{session}_*.MP4",
        f"{subject_id}_{session}_*.avi",
        f"{subject_id}_{session}_*.AVI",
    ]

    for pattern in alt_patterns:
        matches = list(session_dir.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            logger.warning(f"Multiple videos found for {subject_id}/{session}: {[m.name for m in matches]}")
            return matches[0]

    return None


def preprocess_raw_to_csv(config: RawDataConfig) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Convert raw label xlsx files and video structure to labels.csv.

    Args:
        config: Preprocessing configuration

    Returns:
        Tuple of (DataFrame with labels, PreprocessingReport)
    """
    report = PreprocessingReport()
    sessions = discover_sessions(config.label_dir)
    report.total_sessions = len(sessions)

    # Count unique subjects
    subjects = set(s["subject_id"] for s in sessions)
    report.total_subjects = len(subjects)

    all_labels = []
    missing_videos = []

    logger.info(f"Discovered {len(sessions)} sessions from {len(subjects)} subjects")

    for session_info in sessions:
        subject_id = session_info["subject_id"]
        session = session_info["session"]
        xlsx_files = session_info["label_files"]

        # Find corresponding video
        video_path = resolve_video_path(
            config.video_dir,
            subject_id,
            session,
            config.video_filename_pattern,
        )

        if video_path is None:
            missing_videos.append(f"{subject_id}/{session}")
            continue

        # Combine label files
        try:
            labels_df, conflicts = combine_label_files(
                xlsx_files,
                config.label_encoding,
                config.conflict_strategy,
            )
            report.conflicts_found += conflicts
        except Exception as e:
            logger.warning(f"Failed to process labels for {subject_id}/{session}: {e}")
            continue

        if len(labels_df) < config.min_labels_per_video:
            report.sparse_videos.append(f"{subject_id}/{session} ({len(labels_df)} labels)")
            if len(labels_df) == 0:
                continue

        # Add metadata columns
        labels_df["video_path"] = str(video_path)
        labels_df["subject_id"] = subject_id
        labels_df["session"] = session

        all_labels.append(labels_df)
        report.labels_per_video[str(video_path)] = len(labels_df)

    report.missing_videos = missing_videos

    if not all_labels:
        logger.error("No valid label data found!")
        return pd.DataFrame(), report

    # Combine all sessions
    final_df = pd.concat(all_labels, ignore_index=True)

    # Reorder columns
    final_df = final_df[["video_path", "second", "label", "subject_id", "session"]]

    # Update report
    report.total_videos = final_df["video_path"].nunique()
    report.total_labels = len(final_df)
    report.label_distribution = final_df["label"].value_counts().to_dict()

    # Ensure output directory exists
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    final_df.to_csv(config.output_csv, index=False)
    logger.info(
        f"Saved {len(final_df)} labels across {report.total_videos} videos "
        f"to {config.output_csv}"
    )

    return final_df, report


def print_preprocessing_report(report: PreprocessingReport) -> None:
    """Print preprocessing statistics."""
    print("\n" + "=" * 60)
    print("PREPROCESSING REPORT")
    print("=" * 60)
    print(f"Subjects discovered:     {report.total_subjects}")
    print(f"Sessions discovered:     {report.total_sessions}")
    print(f"Videos with labels:      {report.total_videos}")
    print(f"Total labeled seconds:   {report.total_labels}")

    if report.label_distribution:
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(report.label_distribution.items()))
        print(f"Label distribution:      {dist_str}")

    if report.conflicts_found > 0:
        print(f"Label conflicts found:   {report.conflicts_found}")

    print(f"\nMissing videos:          {len(report.missing_videos)}")
    if report.missing_videos:
        for v in report.missing_videos[:10]:
            print(f"  - {v}")
        if len(report.missing_videos) > 10:
            print(f"  ... and {len(report.missing_videos) - 10} more")

    if report.sparse_videos:
        print(f"\nSparse videos (few labels): {len(report.sparse_videos)}")
        for v in report.sparse_videos[:5]:
            print(f"  - {v}")
        if len(report.sparse_videos) > 5:
            print(f"  ... and {len(report.sparse_videos) - 5} more")

    print("=" * 60 + "\n")

#!/usr/bin/env python
"""
Compute inter-rater reliability (IRR) for CARE synchrony annotations.

Produces JSON/CSV reports, human-readable summaries, and diagnostic
visualisations.

Usage
-----
    python scripts/compute_irr.py \\
        --label-dir /path/to/label_dir \\
        --output-dir runs/irr_analysis \\
        --labels-csv data/labels.csv

Outputs
-------
    irr_report.json          Full IRR metrics (machine-readable)
    irr_summary.txt          Human-readable summary
    per_session_irr.csv      Per-session IRR table
    difficulty_scores.csv    Per-second difficulty scores
    kappa_distribution.png   Histogram of per-session Cohen's Kappa
    agreement_vs_session.png Agreement rate per session
    difficulty_distribution.png  Distribution of difficulty scores
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- must come before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``synchronai`` is importable
# when running this script directly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from synchronai.evaluation.irr_analysis import (  # noqa: E402
    compute_difficulty_scores,
    compute_full_irr,
    print_irr_report,
)
from synchronai.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ===================================================================
# CLI argument parsing
# ===================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute inter-rater reliability (IRR) for synchrony annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        required=True,
        help=(
            "Root label directory with {subject_id}/{session}/*.xlsx structure. "
            "Sessions with 2+ xlsx files are treated as multi-annotator."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/irr_analysis"),
        help="Directory for output files (default: runs/irr_analysis/).",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to the combined labels.csv produced by raw_to_csv. "
            "If provided, cross-references IRR with the final training labels."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="a:0,s:1",
        help='Label encoding as comma-separated key:value pairs (default: "a:0,s:1").',
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def _parse_encoding(encoding_str: str) -> dict[str, int]:
    """Parse ``"a:0,s:1"`` into ``{"a": 0, "s": 1}``."""
    mapping: dict[str, int] = {}
    for pair in encoding_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(f"Invalid encoding pair: {pair!r}. Expected key:value.")
        key, value = pair.split(":", 1)
        mapping[key.strip()] = int(value.strip())
    return mapping


# ===================================================================
# Serialisation helpers
# ===================================================================

def _make_json_safe(obj: object) -> object:
    """Recursively convert numpy/Path types so ``json.dumps`` works."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ===================================================================
# Visualisations
# ===================================================================

def _plot_kappa_distribution(per_session: list[dict], output_path: Path) -> None:
    """Histogram of per-session mean Cohen's Kappa values."""
    kappas = [
        s["mean_cohens_kappa"]
        for s in per_session
        if not np.isnan(s["mean_cohens_kappa"])
    ]

    if not kappas:
        logger.warning("No valid kappa values to plot -- skipping kappa_distribution.png")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(kappas, bins=20, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Cohen's Kappa", fontsize=12)
    ax.set_ylabel("Number of sessions", fontsize=12)
    ax.set_title("Distribution of Per-Session Cohen's Kappa", fontsize=14)
    ax.axvline(np.mean(kappas), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(kappas):.3f}")
    ax.axvline(np.median(kappas), color="orange", linestyle=":", linewidth=1.5,
               label=f"Median = {np.median(kappas):.3f}")
    ax.legend(fontsize=10)
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def _plot_agreement_vs_session(per_session: list[dict], output_path: Path) -> None:
    """Bar chart of percent agreement per session."""
    valid = [
        s for s in per_session
        if not np.isnan(s["mean_percent_agreement"])
    ]

    if not valid:
        logger.warning("No valid agreement values to plot -- skipping agreement_vs_session.png")
        return

    # Sort by agreement for visual clarity
    valid = sorted(valid, key=lambda s: s["mean_percent_agreement"])
    labels = [f"{s['subject_id']}/{s['session']}" for s in valid]
    agreements = [s["mean_percent_agreement"] for s in valid]

    fig, ax = plt.subplots(figsize=(max(8, len(valid) * 0.4), 6))

    # Colour bars by agreement level
    colours = []
    for a in agreements:
        if a >= 90:
            colours.append("#2CA02C")   # green
        elif a >= 75:
            colours.append("#FFD700")   # gold
        elif a >= 60:
            colours.append("#FF8C00")   # orange
        else:
            colours.append("#D62728")   # red

    bars = ax.barh(range(len(labels)), agreements, color=colours, edgecolor="black",
                   alpha=0.85)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Percent Agreement", fontsize=12)
    ax.set_title("Agreement Rate per Session", fontsize=14)
    ax.set_xlim(0, 105)

    # Reference lines
    ax.axvline(90, color="green", linestyle="--", alpha=0.5, label="90%")
    ax.axvline(75, color="orange", linestyle="--", alpha=0.5, label="75%")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def _plot_difficulty_distribution(difficulty_df: pd.DataFrame, output_path: Path) -> None:
    """Histogram of difficulty scores."""
    if difficulty_df.empty:
        logger.warning("No difficulty scores to plot -- skipping difficulty_distribution.png")
        return

    scores = difficulty_df["difficulty_score"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax = axes[0]
    ax.hist(scores, bins=20, edgecolor="black", alpha=0.75, color="#DD8452")
    ax.set_xlabel("Difficulty Score", fontsize=12)
    ax.set_ylabel("Number of seconds", fontsize=12)
    ax.set_title("Distribution of Difficulty Scores", fontsize=14)
    ax.axvline(np.mean(scores), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(scores):.3f}")
    ax.legend(fontsize=10)

    # Right: breakdown by consensus label
    ax = axes[1]
    for label_val in sorted(difficulty_df["consensus_label"].unique()):
        subset = difficulty_df[difficulty_df["consensus_label"] == label_val]
        label_name = {0: "Async (0)", 1: "Sync (1)"}.get(label_val, str(label_val))
        ax.hist(
            subset["difficulty_score"].values,
            bins=20,
            alpha=0.6,
            edgecolor="black",
            label=f"{label_name} (n={len(subset)})",
        )
    ax.set_xlabel("Difficulty Score", fontsize=12)
    ax.set_ylabel("Number of seconds", fontsize=12)
    ax.set_title("Difficulty by Consensus Label", fontsize=14)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


# ===================================================================
# Cross-reference with labels.csv
# ===================================================================

def _cross_reference(
    difficulty_df: pd.DataFrame,
    labels_csv_path: Path,
    output_dir: Path,
) -> None:
    """Cross-reference difficulty scores with the final training labels.csv."""
    if not labels_csv_path.exists():
        logger.warning("labels-csv %s does not exist -- skipping cross-reference", labels_csv_path)
        return

    labels_df = pd.read_csv(labels_csv_path)
    logger.info("Loaded labels.csv with %d rows", len(labels_df))

    if difficulty_df.empty:
        logger.warning("No difficulty scores to cross-reference")
        return

    # Merge on subject_id + session + second
    merged = labels_df.merge(
        difficulty_df,
        on=["subject_id", "session", "second"],
        how="inner",
    )

    if merged.empty:
        logger.warning("No overlapping rows between labels.csv and difficulty scores")
        return

    # Check how often the training label matches the consensus
    merged["label_matches_consensus"] = merged["label"] == merged["consensus_label"]
    match_rate = merged["label_matches_consensus"].mean() * 100

    # Summarise
    summary_lines = [
        "",
        "=" * 60,
        "CROSS-REFERENCE: labels.csv vs IRR consensus",
        "=" * 60,
        f"Matched rows:            {len(merged)}",
        f"Label matches consensus: {match_rate:.1f}%",
        f"Mean difficulty (all):   {merged['difficulty_score'].mean():.4f}",
    ]

    mismatches = merged[~merged["label_matches_consensus"]]
    if not mismatches.empty:
        summary_lines.append(
            f"Mismatches:              {len(mismatches)} "
            f"({100 * len(mismatches) / len(merged):.1f}%)"
        )
        summary_lines.append(
            f"Mean difficulty (mismatch): {mismatches['difficulty_score'].mean():.4f}"
        )

    summary_lines.append("=" * 60)
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save cross-reference details
    xref_path = output_dir / "cross_reference.csv"
    merged.to_csv(xref_path, index=False)
    logger.info("Saved cross-reference to %s", xref_path)


# ===================================================================
# Per-session CSV export
# ===================================================================

def _export_per_session_csv(per_session: list[dict], output_path: Path) -> None:
    """Export per-session IRR metrics to a CSV file."""
    rows = []
    for sess in per_session:
        total_conflicts = sum(
            p.get("n_conflicts", 0) for p in sess.get("pairwise", [])
        )
        rows.append(
            {
                "subject_id": sess["subject_id"],
                "session": sess["session"],
                "n_annotators": sess["n_annotators"],
                "mean_cohens_kappa": sess["mean_cohens_kappa"],
                "mean_percent_agreement": sess["mean_percent_agreement"],
                "mean_pabak": sess["mean_pabak"],
                "fleiss_kappa": sess.get("fleiss_kappa"),
                "total_common_seconds": sess["total_common_seconds"],
                "all_seconds": sess["all_seconds"],
                "total_conflicts": total_conflicts,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Saved per-session IRR to %s (%d rows)", output_path, len(df))


# ===================================================================
# Main
# ===================================================================

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    encoding = _parse_encoding(args.encoding)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = args.label_dir
    if not label_dir.exists():
        logger.error("Label directory does not exist: %s", label_dir)
        sys.exit(1)

    # ---- 1. Compute full IRR ----
    print()
    print("=" * 60)
    print("Computing Inter-Rater Reliability (IRR) ...")
    print("=" * 60)
    print(f"  Label directory: {label_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Encoding: {encoding}")
    print()

    report = compute_full_irr(label_dir, encoding=encoding)

    # ---- 2. Print summary to stdout ----
    summary_text = print_irr_report(report)

    # ---- 3. Save JSON report ----
    json_path = output_dir / "irr_report.json"
    safe_report = _make_json_safe(report)
    with open(json_path, "w") as f:
        json.dump(safe_report, f, indent=2)
    logger.info("Saved IRR report to %s", json_path)

    # ---- 4. Save human-readable summary ----
    txt_path = output_dir / "irr_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text)
    logger.info("Saved summary to %s", txt_path)

    # ---- 5. Export per-session CSV ----
    per_session_path = output_dir / "per_session_irr.csv"
    _export_per_session_csv(report.get("per_session", []), per_session_path)

    # ---- 6. Compute and save difficulty scores ----
    print()
    print("Computing difficulty scores ...")
    difficulty_df = compute_difficulty_scores(label_dir, encoding=encoding)
    diff_path = output_dir / "difficulty_scores.csv"
    difficulty_df.to_csv(diff_path, index=False)
    logger.info("Saved difficulty scores to %s (%d rows)", diff_path, len(difficulty_df))

    # ---- 7. Visualisations ----
    print("Generating visualisations ...")
    per_session = report.get("per_session", [])

    _plot_kappa_distribution(per_session, output_dir / "kappa_distribution.png")
    _plot_agreement_vs_session(per_session, output_dir / "agreement_vs_session.png")
    _plot_difficulty_distribution(difficulty_df, output_dir / "difficulty_distribution.png")

    # ---- 8. Cross-reference with labels.csv (optional) ----
    if args.labels_csv is not None:
        print()
        print("Cross-referencing with labels.csv ...")
        _cross_reference(difficulty_df, args.labels_csv, output_dir)

    # ---- Done ----
    print()
    print("=" * 60)
    print(f"IRR analysis complete. Results saved to: {output_dir}/")
    print("=" * 60)
    print(f"  irr_report.json            - Full metrics (JSON)")
    print(f"  irr_summary.txt            - Human-readable summary")
    print(f"  per_session_irr.csv        - Per-session metrics table")
    print(f"  difficulty_scores.csv      - Per-second difficulty scores")
    print(f"  kappa_distribution.png     - Kappa histogram")
    print(f"  agreement_vs_session.png   - Agreement per session")
    print(f"  difficulty_distribution.png - Difficulty score distribution")
    if args.labels_csv is not None:
        print(f"  cross_reference.csv        - IRR x training label comparison")
    print()


if __name__ == "__main__":
    main()

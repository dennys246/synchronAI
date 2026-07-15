"""
Inter-rater reliability (IRR) analysis for synchrony annotations.

This module computes agreement statistics across multiple human annotators
for the CARE synchrony study.  It reuses the flexible xlsx-parsing logic
from ``synchronai.data.preprocessing.raw_to_csv`` so that annotator-level
labels are loaded identically to the combined pipeline.

Typical workflow
----------------
1.  ``discover_annotator_pairs``  -- find sessions with 2+ annotator files
2.  ``compute_full_irr``          -- run per-session and aggregate analysis
3.  ``print_irr_report``          -- display results
4.  ``compute_difficulty_scores`` -- per-second disagreement scores

Metrics produced
~~~~~~~~~~~~~~~~
- Cohen's Kappa (pairwise, via sklearn)
- Percent agreement (pairwise)
- PABAK -- Prevalence-Adjusted Bias-Adjusted Kappa (pairwise)
- Fleiss' Kappa (multi-rater, implemented from scratch)
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from synchronai.data.preprocessing.raw_to_csv import (
    load_label_xlsx,
    _filter_duration_markers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Loading individual annotator labels
# ---------------------------------------------------------------------------


def load_annotator_labels(
    xlsx_path: Path,
    encoding: dict[str, int] | None = None,
) -> dict[int, int]:
    """Load a single annotator's labels from an xlsx file.

    Delegates to the flexible parsing pipeline in
    ``synchronai.data.preprocessing.raw_to_csv.load_label_xlsx``, which
    tries simple two-column format, named columns, heuristic detection,
    and headerless scanning.

    Args:
        xlsx_path: Path to the xlsx label file.
        encoding: Mapping of raw label codes to integers.
            Defaults to ``{"a": 0, "s": 1}``.

    Returns:
        Dictionary mapping ``second`` (int) to ``label`` (int).

    Raises:
        ValueError: If the file cannot be parsed into valid labels.
    """
    if encoding is None:
        encoding = {"a": 0, "s": 1}

    xlsx_path = Path(xlsx_path)
    df = load_label_xlsx(xlsx_path, encoding)
    df = _filter_duration_markers(df)

    if df.empty:
        logger.warning("No labels loaded from %s", xlsx_path)
        return {}

    # Deduplicate: keep last occurrence per second (matches raw_to_csv)
    df = df.drop_duplicates(subset="second", keep="last")

    return dict(zip(df["second"].astype(int), df["label"].astype(int)))


# ---------------------------------------------------------------------------
# 2. Discovering annotator pairs
# ---------------------------------------------------------------------------


def extract_annotator_id(xlsx_path: Path | str) -> str:
    """Extract an annotator identifier from an xlsx filename.

    The naming convention is ``{subject_id}_{session}_{activity}.xlsx``.
    The *activity* segment acts as the annotator / annotation-round
    identifier.  If the filename has fewer than three underscore-separated
    parts, the full stem (without extension) is returned.

    Args:
        xlsx_path: Path to the xlsx file.

    Returns:
        String identifier for the annotator.
    """
    stem = Path(xlsx_path).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        # Everything after subject_id and session
        return "_".join(parts[2:])
    return stem


def discover_annotator_pairs(label_dir: Path | str) -> list[dict[str, Any]]:
    """Discover sessions that have multiple annotator xlsx files.

    Walks the ``{label_dir}/{subject_id}/{session}/`` directory tree and
    returns only those sessions where two or more valid xlsx files exist
    (excluding macOS resource-fork artefacts and Excel temp files).

    Args:
        label_dir: Root label directory.

    Returns:
        List of dicts, each containing:
            - ``subject_id`` (str)
            - ``session`` (str)
            - ``annotator_files`` (list[Path])
            - ``annotator_ids`` (list[str])
    """
    label_dir = Path(label_dir)
    results: list[dict[str, Any]] = []

    if not label_dir.exists():
        logger.error("Label directory does not exist: %s", label_dir)
        return results

    for subject_path in sorted(label_dir.iterdir()):
        if not subject_path.is_dir() or subject_path.name.startswith("."):
            continue

        subject_id = subject_path.name

        for session_path in sorted(subject_path.iterdir()):
            if not session_path.is_dir() or session_path.name.startswith("."):
                continue

            session = session_path.name

            xlsx_files = sorted(
                f
                for f in session_path.glob("*.xlsx")
                if not f.name.startswith("~$") and not f.name.startswith("._")
            )

            if len(xlsx_files) >= 2:
                annotator_ids = [extract_annotator_id(f) for f in xlsx_files]
                results.append(
                    {
                        "subject_id": subject_id,
                        "session": session,
                        "annotator_files": xlsx_files,
                        "annotator_ids": annotator_ids,
                    }
                )

    logger.info(
        "Discovered %d sessions with multiple annotators across %d subjects",
        len(results),
        len({r["subject_id"] for r in results}),
    )
    return results


# ---------------------------------------------------------------------------
# 3. Pairwise IRR computation
# ---------------------------------------------------------------------------


def compute_pairwise_irr(
    labels_a: dict[int, int],
    labels_b: dict[int, int],
) -> dict[str, Any]:
    """Compute pairwise inter-rater reliability between two annotators.

    Only seconds present in *both* annotators' label sets are compared.

    Args:
        labels_a: Mapping of ``second -> label`` for annotator A.
        labels_b: Mapping of ``second -> label`` for annotator B.

    Returns:
        Dictionary with keys:
            - ``cohens_kappa`` (float)
            - ``percent_agreement`` (float, 0--100)
            - ``pabak`` (float) -- Prevalence-Adjusted Bias-Adjusted Kappa
            - ``n_common`` (int) -- number of overlapping seconds
            - ``n_conflicts`` (int) -- seconds where annotators disagree
            - ``conflict_seconds`` (list[int]) -- the disagreeing seconds
    """
    common_seconds = sorted(set(labels_a.keys()) & set(labels_b.keys()))

    if not common_seconds:
        logger.warning("No overlapping seconds between annotator pair")
        return {
            "cohens_kappa": float("nan"),
            "percent_agreement": float("nan"),
            "pabak": float("nan"),
            "n_common": 0,
            "n_conflicts": 0,
            "conflict_seconds": [],
        }

    y_a = np.array([labels_a[s] for s in common_seconds])
    y_b = np.array([labels_b[s] for s in common_seconds])

    n = len(common_seconds)
    agree = int((y_a == y_b).sum())
    disagree_mask = y_a != y_b
    conflicts = [common_seconds[i] for i in range(n) if disagree_mask[i]]
    percent_agreement = 100.0 * agree / n

    # Cohen's Kappa (sklearn handles edge cases like constant raters)
    try:
        kappa = float(cohen_kappa_score(y_a, y_b))
    except Exception:
        kappa = float("nan")

    # PABAK = 2 * p_o - 1  (where p_o = observed agreement proportion)
    p_o = agree / n
    pabak = 2.0 * p_o - 1.0

    return {
        "cohens_kappa": kappa,
        "percent_agreement": percent_agreement,
        "pabak": pabak,
        "n_common": n,
        "n_conflicts": len(conflicts),
        "conflict_seconds": conflicts,
    }


# ---------------------------------------------------------------------------
# 4. Session-level IRR (including Fleiss' Kappa)
# ---------------------------------------------------------------------------


def _fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' Kappa for multiple raters.

    Args:
        ratings_matrix: 2-D array of shape ``(n_subjects, n_categories)``
            where ``ratings_matrix[i, j]`` is the number of raters who
            assigned subject *i* to category *j*.

    Returns:
        Fleiss' Kappa (float).  Returns ``nan`` if undefined.
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = int(ratings_matrix[0].sum())

    if n_raters <= 1 or n_subjects == 0:
        return float("nan")

    # Proportion of all assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement extent
    P_i = (
        (ratings_matrix ** 2).sum(axis=1) - n_raters
    ) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if abs(1.0 - P_e) < 1e-10:
        return float("nan")

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(kappa)


def compute_session_irr(
    annotator_labels_list: list[tuple[str, dict[int, int]]],
) -> dict[str, Any]:
    """Compute IRR metrics for a single session with multiple annotators.

    Args:
        annotator_labels_list: List of ``(annotator_id, labels_dict)`` tuples
            where ``labels_dict`` maps ``second -> label``.

    Returns:
        Dictionary with:
            - ``n_annotators`` (int)
            - ``annotator_ids`` (list[str])
            - ``pairwise`` (list[dict]) -- one entry per annotator pair
            - ``mean_cohens_kappa`` (float)
            - ``mean_percent_agreement`` (float)
            - ``mean_pabak`` (float)
            - ``fleiss_kappa`` (float or None if <= 2 annotators)
            - ``total_common_seconds`` (int) -- seconds labelled by all
            - ``all_seconds`` (int) -- union of all labelled seconds
    """
    n = len(annotator_labels_list)
    ids = [aid for aid, _ in annotator_labels_list]

    # ------ pairwise metrics ------
    pairwise_results: list[dict[str, Any]] = []
    for (id_a, lab_a), (id_b, lab_b) in combinations(annotator_labels_list, 2):
        result = compute_pairwise_irr(lab_a, lab_b)
        result["annotator_a"] = id_a
        result["annotator_b"] = id_b
        pairwise_results.append(result)

    kappas = [r["cohens_kappa"] for r in pairwise_results if not np.isnan(r["cohens_kappa"])]
    agreements = [r["percent_agreement"] for r in pairwise_results if not np.isnan(r["percent_agreement"])]
    pabaks = [r["pabak"] for r in pairwise_results if not np.isnan(r["pabak"])]

    mean_kappa = float(np.mean(kappas)) if kappas else float("nan")
    mean_agreement = float(np.mean(agreements)) if agreements else float("nan")
    mean_pabak = float(np.mean(pabaks)) if pabaks else float("nan")

    # ------ union / intersection counts ------
    all_seconds_set: set[int] = set()
    for _, lab in annotator_labels_list:
        all_seconds_set.update(lab.keys())

    common_seconds_set = set.intersection(*(set(lab.keys()) for _, lab in annotator_labels_list)) if n > 0 else set()

    # ------ Fleiss' Kappa (only meaningful for > 2 raters) ------
    fleiss = None
    if n > 2 and common_seconds_set:
        all_labels = set()
        for _, lab in annotator_labels_list:
            all_labels.update(lab.values())
        categories = sorted(all_labels)
        cat_index = {c: i for i, c in enumerate(categories)}

        seconds_sorted = sorted(common_seconds_set)
        matrix = np.zeros((len(seconds_sorted), len(categories)), dtype=int)
        for _, lab in annotator_labels_list:
            for row_idx, sec in enumerate(seconds_sorted):
                if sec in lab:
                    matrix[row_idx, cat_index[lab[sec]]] += 1

        fleiss = _fleiss_kappa(matrix)

    return {
        "n_annotators": n,
        "annotator_ids": ids,
        "pairwise": pairwise_results,
        "mean_cohens_kappa": mean_kappa,
        "mean_percent_agreement": mean_agreement,
        "mean_pabak": mean_pabak,
        "fleiss_kappa": fleiss,
        "total_common_seconds": len(common_seconds_set),
        "all_seconds": len(all_seconds_set),
    }


# ---------------------------------------------------------------------------
# 5. Full IRR analysis
# ---------------------------------------------------------------------------


def compute_full_irr(
    label_dir: Path | str,
    encoding: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run the complete inter-rater reliability analysis.

    This is the main entry point.  It discovers all sessions with multiple
    annotators, computes per-session IRR, and aggregates to overall metrics.

    Args:
        label_dir: Root label directory with
            ``{subject_id}/{session}/*.xlsx`` structure.
        encoding: Label code mapping (default ``{"a": 0, "s": 1}``).

    Returns:
        Comprehensive IRR report dictionary with keys:
            - ``per_session`` (list[dict])  -- per-session IRR summaries
            - ``overall`` (dict)            -- aggregated statistics
            - ``n_sessions_analysed`` (int)
            - ``n_sessions_skipped`` (int)
    """
    if encoding is None:
        encoding = {"a": 0, "s": 1}

    label_dir = Path(label_dir)
    sessions = discover_annotator_pairs(label_dir)

    per_session: list[dict[str, Any]] = []
    n_skipped = 0

    for sess in sessions:
        subject_id = sess["subject_id"]
        session = sess["session"]
        files = sess["annotator_files"]
        ann_ids = sess["annotator_ids"]

        logger.info(
            "Computing IRR for %s/%s (%d annotators)",
            subject_id,
            session,
            len(files),
        )

        # Load each annotator's labels
        annotator_labels: list[tuple[str, dict[int, int]]] = []
        for fpath, aid in zip(files, ann_ids):
            try:
                labels = load_annotator_labels(fpath, encoding)
                if labels:
                    annotator_labels.append((aid, labels))
                else:
                    logger.warning(
                        "Empty labels for annotator %s in %s/%s",
                        aid, subject_id, session,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load %s for %s/%s: %s",
                    fpath.name, subject_id, session, exc,
                )

        if len(annotator_labels) < 2:
            logger.warning(
                "Fewer than 2 valid annotators for %s/%s -- skipping",
                subject_id, session,
            )
            n_skipped += 1
            continue

        session_irr = compute_session_irr(annotator_labels)
        session_irr["subject_id"] = subject_id
        session_irr["session"] = session
        per_session.append(session_irr)

    # ---- aggregate ----
    overall = _aggregate_irr(per_session)

    report = {
        "per_session": per_session,
        "overall": overall,
        "n_sessions_analysed": len(per_session),
        "n_sessions_skipped": n_skipped,
    }

    logger.info(
        "IRR analysis complete: %d sessions analysed, %d skipped",
        len(per_session),
        n_skipped,
    )
    return report


def _aggregate_irr(per_session: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-session IRR metrics into overall statistics."""
    if not per_session:
        return {
            "mean_cohens_kappa": float("nan"),
            "std_cohens_kappa": float("nan"),
            "median_cohens_kappa": float("nan"),
            "min_cohens_kappa": float("nan"),
            "max_cohens_kappa": float("nan"),
            "mean_percent_agreement": float("nan"),
            "std_percent_agreement": float("nan"),
            "mean_pabak": float("nan"),
            "std_pabak": float("nan"),
            "total_annotated_seconds": 0,
            "total_common_seconds": 0,
            "n_sessions": 0,
        }

    kappas = [s["mean_cohens_kappa"] for s in per_session if not np.isnan(s["mean_cohens_kappa"])]
    agreements = [s["mean_percent_agreement"] for s in per_session if not np.isnan(s["mean_percent_agreement"])]
    pabaks = [s["mean_pabak"] for s in per_session if not np.isnan(s["mean_pabak"])]
    total_all = sum(s["all_seconds"] for s in per_session)
    total_common = sum(s["total_common_seconds"] for s in per_session)

    fleiss_vals = [
        s["fleiss_kappa"] for s in per_session
        if s["fleiss_kappa"] is not None and not np.isnan(s["fleiss_kappa"])
    ]

    overall: dict[str, Any] = {
        "mean_cohens_kappa": float(np.mean(kappas)) if kappas else float("nan"),
        "std_cohens_kappa": float(np.std(kappas)) if kappas else float("nan"),
        "median_cohens_kappa": float(np.median(kappas)) if kappas else float("nan"),
        "min_cohens_kappa": float(np.min(kappas)) if kappas else float("nan"),
        "max_cohens_kappa": float(np.max(kappas)) if kappas else float("nan"),
        "mean_percent_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "std_percent_agreement": float(np.std(agreements)) if agreements else float("nan"),
        "mean_pabak": float(np.mean(pabaks)) if pabaks else float("nan"),
        "std_pabak": float(np.std(pabaks)) if pabaks else float("nan"),
        "total_annotated_seconds": total_all,
        "total_common_seconds": total_common,
        "n_sessions": len(per_session),
    }

    if fleiss_vals:
        overall["mean_fleiss_kappa"] = float(np.mean(fleiss_vals))
        overall["std_fleiss_kappa"] = float(np.std(fleiss_vals))

    return overall


# ---------------------------------------------------------------------------
# 6. Difficulty scores
# ---------------------------------------------------------------------------


def compute_difficulty_scores(
    label_dir: Path | str,
    encoding: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute per-second difficulty scores based on annotator disagreement.

    The difficulty score for a given second is the fraction of annotators
    who *disagree* with the consensus (majority) label.  A score of 0
    means perfect agreement; a score of 0.5 means a 50/50 split.

    Args:
        label_dir: Root label directory.
        encoding: Label code mapping (default ``{"a": 0, "s": 1}``).

    Returns:
        DataFrame with columns:
            - ``subject_id``
            - ``session``
            - ``second``
            - ``consensus_label`` (int)
            - ``difficulty_score`` (float, 0.0--0.5)
            - ``n_annotators`` (int)
    """
    if encoding is None:
        encoding = {"a": 0, "s": 1}

    label_dir = Path(label_dir)
    sessions = discover_annotator_pairs(label_dir)

    rows: list[dict[str, Any]] = []

    for sess in sessions:
        subject_id = sess["subject_id"]
        session = sess["session"]
        files = sess["annotator_files"]

        # Load all annotators
        all_labels: list[dict[int, int]] = []
        for fpath in files:
            try:
                labels = load_annotator_labels(fpath, encoding)
                if labels:
                    all_labels.append(labels)
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath.name, exc)

        if len(all_labels) < 2:
            continue

        # Collect every labelled second across all annotators
        all_seconds: set[int] = set()
        for lab in all_labels:
            all_seconds.update(lab.keys())

        for sec in sorted(all_seconds):
            votes = [lab[sec] for lab in all_labels if sec in lab]
            n = len(votes)
            if n < 2:
                continue

            # Consensus = majority vote (ties broken by label value 1/sync)
            vote_counts: dict[int, int] = {}
            for v in votes:
                vote_counts[v] = vote_counts.get(v, 0) + 1
            consensus = max(vote_counts, key=lambda k: (vote_counts[k], k))

            n_disagree = sum(1 for v in votes if v != consensus)
            difficulty = n_disagree / n

            rows.append(
                {
                    "subject_id": subject_id,
                    "session": session,
                    "second": sec,
                    "consensus_label": consensus,
                    "difficulty_score": difficulty,
                    "n_annotators": n,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No difficulty scores computed -- no multi-annotator data found")
        return pd.DataFrame(
            columns=[
                "subject_id",
                "session",
                "second",
                "consensus_label",
                "difficulty_score",
                "n_annotators",
            ]
        )

    logger.info(
        "Computed difficulty scores for %d seconds across %d sessions",
        len(df),
        df.groupby(["subject_id", "session"]).ngroups,
    )
    return df


# ---------------------------------------------------------------------------
# 7. Pretty-printing
# ---------------------------------------------------------------------------


def print_irr_report(report: dict[str, Any]) -> str:
    """Pretty-print IRR results and return the formatted string.

    Args:
        report: Dictionary returned by :func:`compute_full_irr`.

    Returns:
        The formatted report as a string (also printed to stdout).
    """
    lines: list[str] = []

    def _add(text: str = "") -> None:
        lines.append(text)

    overall = report.get("overall", {})
    per_session = report.get("per_session", [])

    _add()
    _add("=" * 70)
    _add("INTER-RATER RELIABILITY (IRR) REPORT")
    _add("=" * 70)
    _add()

    _add(f"Sessions analysed:          {report.get('n_sessions_analysed', 0)}")
    _add(f"Sessions skipped:           {report.get('n_sessions_skipped', 0)}")
    _add(f"Total annotated seconds:    {overall.get('total_annotated_seconds', 0)}")
    _add(f"Total common seconds:       {overall.get('total_common_seconds', 0)}")
    _add()

    _add("-" * 70)
    _add("OVERALL METRICS (averaged across sessions)")
    _add("-" * 70)

    def _fmt(val: Any) -> str:
        if val is None:
            return "N/A"
        try:
            if np.isnan(val):
                return "N/A"
        except (TypeError, ValueError):
            pass
        return f"{val:.4f}"

    _add(f"  Cohen's Kappa:   mean={_fmt(overall.get('mean_cohens_kappa'))}"
         f"  std={_fmt(overall.get('std_cohens_kappa'))}"
         f"  median={_fmt(overall.get('median_cohens_kappa'))}")
    _add(f"                   min={_fmt(overall.get('min_cohens_kappa'))}"
         f"   max={_fmt(overall.get('max_cohens_kappa'))}")
    _add(f"  Percent agree:   mean={_fmt(overall.get('mean_percent_agreement'))}%"
         f"  std={_fmt(overall.get('std_percent_agreement'))}%")
    _add(f"  PABAK:           mean={_fmt(overall.get('mean_pabak'))}"
         f"  std={_fmt(overall.get('std_pabak'))}")

    if "mean_fleiss_kappa" in overall:
        _add(f"  Fleiss' Kappa:   mean={_fmt(overall.get('mean_fleiss_kappa'))}"
             f"  std={_fmt(overall.get('std_fleiss_kappa'))}")

    _add()

    # Interpretation guide
    _add("-" * 70)
    _add("INTERPRETATION GUIDE (Landis & Koch, 1977)")
    _add("-" * 70)
    _add("  < 0.00  Poor")
    _add("  0.00 - 0.20  Slight")
    _add("  0.21 - 0.40  Fair")
    _add("  0.41 - 0.60  Moderate")
    _add("  0.61 - 0.80  Substantial")
    _add("  0.81 - 1.00  Almost perfect")
    _add()

    # Per-session breakdown
    if per_session:
        _add("-" * 70)
        _add("PER-SESSION BREAKDOWN")
        _add("-" * 70)

        header = (
            f"{'Subject':<12} {'Session':<8} {'# Ann':>5} "
            f"{'Kappa':>8} {'Agree%':>8} {'PABAK':>8} "
            f"{'Common':>8} {'Conflicts':>10}"
        )
        _add(header)
        _add("-" * len(header))

        for sess in per_session:
            total_conflicts = sum(
                p.get("n_conflicts", 0) for p in sess.get("pairwise", [])
            )
            _add(
                f"{sess['subject_id']:<12} {sess['session']:<8} "
                f"{sess['n_annotators']:>5} "
                f"{_fmt(sess['mean_cohens_kappa']):>8} "
                f"{_fmt(sess['mean_percent_agreement']):>8} "
                f"{_fmt(sess['mean_pabak']):>8} "
                f"{sess['total_common_seconds']:>8} "
                f"{total_conflicts:>10}"
            )

    _add()
    _add("=" * 70)
    _add()

    text = "\n".join(lines)
    print(text)
    return text

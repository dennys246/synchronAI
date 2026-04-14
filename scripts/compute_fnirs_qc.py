#!/usr/bin/env python3
"""Pre-compute fNIRS quality control tiers for all recordings.

Runs QC (SCI, cardiac, SNR) on every recording and writes a CSV cache:
    fnirs_path, quality_tier, mean_sci, scan_snr, n_pairs_passed, n_pairs_total

This runs independently of feature extraction — no encoder needed.
The extraction script can then read this cache via --qc-cache instead of
running QC inline (which doubles load time per recording).

Usage:
    python scripts/compute_fnirs_qc.py \
        --data-dirs "/path/to/NIRS_data:/path/to/more" \
        --output qc_tiers.csv \
        --sci-threshold 0.40 \
        --snr-threshold 2.0
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute fNIRS quality tiers")
    parser.add_argument("--data-dirs", required=True,
                        help="Colon-separated fNIRS data directories")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g., data/qc_tiers.csv)")
    parser.add_argument("--sci-threshold", type=float, default=0.40)
    parser.add_argument("--snr-threshold", type=float, default=2.0)
    parser.add_argument("--cardiac-peak-ratio", type=float, default=2.0)
    parser.add_argument("--no-require-cardiac", action="store_true")
    parser.add_argument("--signal-type", default="hemodynamic",
                        choices=["hemodynamic", "neural"])

    args = parser.parse_args()

    from synchronai.data.fnirs.processing import read_raw_fnirs, load_fnirs
    from synchronai.data.fnirs.quality_control import run_quality_control

    # Reuse discovery logic from extraction script
    from extract_fnirs_features import discover_fnirs_paths

    fnirs_paths = discover_fnirs_paths(args.data_dirs, args.signal_type)
    if not fnirs_paths:
        logger.error("No fNIRS recordings found!")
        return

    logger.info(f"Computing QC tiers for {len(fnirs_paths)} recordings...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_success = 0
    n_fail = 0
    start_time = time.time()

    for i, fnirs_path in enumerate(fnirs_paths):
        try:
            raw_scan = read_raw_fnirs(fnirs_path)
            preprocessed = load_fnirs(fnirs_path, deconvolution=False)
            qc_report = run_quality_control(
                raw_scan,
                preprocessed,
                sci_threshold=args.sci_threshold,
                snr_threshold=args.snr_threshold,
                cardiac_peak_ratio=args.cardiac_peak_ratio,
                require_cardiac=not args.no_require_cardiac,
            )
            del raw_scan, preprocessed

            mean_sci = 0.0
            if qc_report.channel_sci:
                import numpy as np
                vals = list(qc_report.channel_sci.values())
                if vals:
                    mean_sci = float(np.mean(vals))

            rows.append({
                "fnirs_path": fnirs_path,
                "quality_tier": qc_report.quality_tier,
                "mean_sci": f"{mean_sci:.4f}",
                "scan_snr": f"{qc_report.scan_snr:.4f}" if qc_report.scan_snr is not None else "",
                "n_pairs_passed": qc_report.n_channels_after,
                "n_pairs_total": qc_report.n_channels_before,
                "scan_passed": qc_report.scan_passed,
            })
            n_success += 1

        except Exception as e:
            logger.warning(f"QC failed for {fnirs_path}: {e}")
            rows.append({
                "fnirs_path": fnirs_path,
                "quality_tier": "failed",
                "mean_sci": "",
                "scan_snr": "",
                "n_pairs_passed": 0,
                "n_pairs_total": 0,
                "scan_passed": False,
            })
            n_fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(fnirs_paths) - i - 1) / rate / 60
            logger.info(
                f"  {i+1}/{len(fnirs_paths)} ({rate:.1f}/sec, ETA: {eta:.0f} min) "
                f"— {n_success} ok, {n_fail} failed"
            )

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    elapsed = time.time() - start_time
    logger.info(f"QC complete in {elapsed:.1f}s")
    logger.info(f"  Success: {n_success}, Failed: {n_fail}")
    logger.info(f"  Output: {output_path}")

    # Tier distribution
    from collections import Counter
    tier_counts = Counter(r["quality_tier"] for r in rows)
    for tier, count in sorted(tier_counts.items()):
        logger.info(f"  {tier}: {count}")


if __name__ == "__main__":
    main()

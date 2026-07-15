#!/usr/bin/env python3
"""
Pack an existing fNIRS feature directory into a single mmap-friendly binary.

Converts `feature_dir/features/*.pt` (one file per sample) into
`feature_dir/features_packed.bin` + `feature_dir/features_meta.json`,
and appends a `row_idx` column to `feature_dir/feature_index.csv`.

Usage:
    python scripts/pack_features.py <feature_dir>
    python scripts/pack_features.py <feature_dir> --delete-unpacked
"""

from __future__ import annotations

import argparse
import logging
import sys

from synchronai.data.fnirs.feature_dataset import pack_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_dir", help="Directory with feature_index.csv + features/")
    parser.add_argument(
        "--delete-unpacked", action="store_true",
        help="Remove individual .pt files after successful pack (saves disk).",
    )
    args = parser.parse_args()

    stats = pack_features(args.feature_dir, delete_unpacked=args.delete_unpacked)
    print(f"Done: {stats['n_entries']} entries, {stats['size_mb']:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

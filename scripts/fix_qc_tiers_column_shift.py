"""Repair the column-shift in data/qc_tiers.csv that silently dropped R01 data.

A block of 113 R01 recordings (PSU + WUSTL, all timepoints) were written to the QC
cache under an OLDER schema:

    fnirs_path, scan_passed, quality_tier, scan_snr, n_pairs_passed, n_pairs_total, <blank>

instead of the canonical:

    fnirs_path, quality_tier, mean_sci, scan_snr, n_pairs_passed, n_pairs_total, scan_passed

So those rows have a boolean (`True`/`False`) where `quality_tier` should be, and
their REAL tier (gold/standard/salvageable/rejected) sits in the `mean_sci` slot.
Because the tier reads as `True`, `--include-tiers gold,standard` excluded all of
them, dropping ~110 usable R01 recordings from pretraining AND transfer (corpus was
2001; should be ~2111). The loss is 100% R01 — the under-represented site.

This remaps the shifted rows back to the canonical schema (mean_sci is left blank for
them — the old schema never recorded SCI; the tier, SNR, and pair counts are recovered).
Canonical rows pass through untouched, so the script is idempotent. The original is
backed up to <path>.orig before writing.

Detection: a data row is shifted iff field[1] (the quality_tier slot) is a boolean
literal `True`/`False` rather than a tier name.

Run on the CLUSTER (where data/qc_tiers.csv is authoritative):
    python scripts/fix_qc_tiers_column_shift.py --qc-cache data/qc_tiers.csv
    python scripts/fix_qc_tiers_column_shift.py --qc-cache data/qc_tiers.csv --dry-run
"""
import argparse
import csv
import os
import shutil
from collections import Counter

CANONICAL = ["fnirs_path", "quality_tier", "mean_sci", "scan_snr",
             "n_pairs_passed", "n_pairs_total", "scan_passed"]
BOOL_LITERALS = {"True", "False"}


def is_shifted(row):
    """A shifted row has a boolean where quality_tier (field 1) should be."""
    return len(row) >= 3 and row[1] in BOOL_LITERALS


def remap_shifted(row):
    """Old [path, scan_passed, tier, snr, n_pass, n_tot, blank] -> canonical."""
    # pad defensively so indexing is safe
    r = list(row) + [""] * (7 - len(row))
    return [
        r[0],   # fnirs_path
        r[2],   # quality_tier  (the real tier, recovered)
        "",     # mean_sci       (not recorded in the old schema)
        r[3],   # scan_snr
        r[4],   # n_pairs_passed
        r[5],   # n_pairs_total
        r[1],   # scan_passed    (the boolean that was up front)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc-cache", default="data/qc_tiers.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    with open(args.qc_cache, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    if header != CANONICAL:
        print("WARNING: unexpected header, proceeding anyway:\n  %s" % header)

    before = Counter(r[1] for r in data if len(r) > 1)
    fixed, out = 0, []
    recovered_tiers = Counter()
    for r in data:
        if is_shifted(r):
            new = remap_shifted(r)
            recovered_tiers[new[1]] += 1
            out.append(new)
            fixed += 1
        else:
            out.append(r)
    after = Counter(r[1] for r in out if len(r) > 1)

    print("shifted rows detected/remapped: %d" % fixed)
    print("recovered tiers (now correctly labeled): %s" % dict(recovered_tiers))
    print("quality_tier distribution BEFORE: %s" % dict(before))
    print("quality_tier distribution AFTER:  %s" % dict(after))
    leftover = [t for t in after if t in BOOL_LITERALS]
    assert not leftover, "still have boolean tiers after fix: %s" % leftover

    if args.dry_run:
        print("\n--dry-run: no files written")
        return
    if fixed == 0:
        print("\nnothing to fix (already canonical) — no write")
        return

    backup = args.qc_cache + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(args.qc_cache, backup)
        print("backed up original -> %s" % backup)
    else:
        print("backup already exists (%s) — not overwriting it" % backup)

    tmp = args.qc_cache + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CANONICAL)
        w.writerows(out)
    os.replace(tmp, args.qc_cache)  # atomic
    print("wrote repaired %s (%d data rows)" % (args.qc_cache, len(out)))


if __name__ == "__main__":
    main()

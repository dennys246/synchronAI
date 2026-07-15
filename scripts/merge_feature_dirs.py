"""Merge several extracted feature dirs into one, for combined-training.

`train_multimodal_from_features.py` takes a single --video-feature-dir /
--audio-feature-dir, each a directory with `feature_index.csv` + `features/*.pt`.
To train on CARE/P-CAT + a new study, the new study's features must live in one
combined dir alongside the existing ones.

`feature_file` values are bare filenames whose hash is derived from the video
path, so they never collide across studies. We hard-link (fall back to symlink,
then copy) each .pt into the combined `features/` dir and concatenate the
index files -- no re-extraction, near-zero disk cost.

Usage:
  python scripts/merge_feature_dirs.py \
      --out data/dinov2_features_meanpatch_train_plus_r56 \
      data/dinov2_features_meanpatch data/dinov2_features_meanpatch_r56pcat

Run once per modality (video dirs together, audio dirs together). Keep the
video and audio member ordering identical so the same rows exist in both.
"""

from __future__ import annotations

import argparse
import csv
import os


def link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="combined output feature dir")
    ap.add_argument("dirs", nargs="+", help="member feature dirs to merge (>=2)")
    args = ap.parse_args()

    out_feat = os.path.join(args.out, "features")
    os.makedirs(out_feat, exist_ok=True)

    # Reconcile on the columns COMMON to every member, ordered by the first
    # member's header. Members may carry extra metadata columns that differ
    # across extraction versions (e.g. the newer WavLM extractor adds
    # `n_layers`/`all_layers`); these don't describe the feature tensors and
    # aren't read by train_multimodal_from_features, so we drop them rather than
    # reject the merge. The tensors themselves must still match — feature_dim /
    # n_frames are among the required columns and are asserted equal downstream.
    idx_paths = [os.path.join(d, "feature_index.csv") for d in args.dirs]
    for idx in idx_paths:
        if not os.path.isfile(idx):
            raise SystemExit(f"missing {idx}")
    headers = []
    for idx in idx_paths:
        with open(idx, newline="") as fh:
            headers.append(next(csv.reader(fh)))
    common = set(headers[0])
    for h in headers[1:]:
        common &= set(h)
    out_cols = [c for c in headers[0] if c in common]
    required = {"feature_file", "video_path", "second", "label",
                "subject_id", "feature_dim", "n_frames"}
    missing = required - set(out_cols)
    if missing:
        raise SystemExit(
            f"columns {sorted(missing)} not common to all members; "
            f"headers differ beyond cosmetic metadata:\n  "
            + "\n  ".join(str(h) for h in headers)
        )
    dropped = set().union(*(set(h) for h in headers)) - common
    if dropped:
        print(f"  reconciled on {len(out_cols)} common columns; dropped {sorted(dropped)}")

    rows: list[list[str]] = []
    seen: set[str] = set()
    for d in args.dirs:
        idx = os.path.join(d, "feature_index.csv")
        feat = os.path.join(d, "features")
        with open(idx, newline="") as fh:
            r = csv.DictReader(fh)
            n = 0
            for row in r:
                fname = row["feature_file"]
                if fname in seen:
                    continue  # same (video,second) already contributed by an earlier dir
                seen.add(fname)
                link_or_copy(os.path.join(feat, fname), os.path.join(out_feat, fname))
                rows.append([row[c] for c in out_cols])
                n += 1
        print(f"  {d}: +{n} rows")

    with open(os.path.join(args.out, "feature_index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(out_cols)
        w.writerows(rows)
    print(f"merged {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

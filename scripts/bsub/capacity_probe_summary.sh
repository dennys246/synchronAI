#!/bin/bash
# Read the capacity probe runs + reference points and print the train_acc curve.
# Run from repo root after both submit_capacity_probe_fold0 jobs finish.

python3 - <<'PY'
import json
from pathlib import Path

runs = [
    ("h=64,  reg=default", "runs/multimodal_features/v2_baseline_v6_cv5/fold_0"),
    ("h=64,  reg=low    ", "runs/multimodal_features/v2_regprobe_fold0_lowdrop"),
    ("h=128, reg=default", "runs/multimodal_features/v2_higher_capacity"),
    ("h=128, reg=low    ", "runs/multimodal_features/v2_caprobe_h128_fold0"),
    ("h=256, reg=low    ", "runs/multimodal_features/v2_caprobe_h256_fold0"),
]

print(f"{'config':22s}  {'epochs':>6s}  {'max_train':>9s}  {'max_val':>7s}  {'best_val_auc':>12s}")
print("-" * 70)
for label, path in runs:
    p = Path(path) / "history.json"
    if not p.exists():
        print(f"{label}  (not yet on disk: {path})")
        continue
    h = json.load(open(p))
    n = len(h["train_accs"])
    mt = max(h["train_accs"])
    mv = max(h["val_accs"])
    ma = max(h["val_aucs"])
    print(f"{label}  {n:>6d}  {mt:>9.3f}  {mv:>7.3f}  {ma:>12.4f}")
PY

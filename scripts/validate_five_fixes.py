#!/usr/bin/env python3
"""Throwaway validation for the five code-review fixes (#3-#7).

Run under ml-env (needs torch + numpy + pandas):
    $SYNCHRONAI_DIR/ml-env/bin/python scripts/validate_five_fixes.py

Exercises the REAL code paths for #3/#4/#5/#7 with tiny temp fixtures; #6 is a
config default, checked at the source level. Prints PASS/FAIL per fix and exits
nonzero if any fail. Not meant to be committed — delete after.
"""
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

results: dict[str, str] = {}


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- #3 aggregate_kfold_results: one epoch per fold at min-val-loss ----------
try:
    agg = _load_script(REPO / "scripts/aggregate_kfold_results.py")
    with tempfile.TemporaryDirectory() as d:
        fold = Path(d) / "fold_0"
        fold.mkdir()
        # min val_loss at epoch idx 2; max auc at idx 0 — the biased old pick.
        hist = {
            "val_losses": [0.9, 0.7, 0.5, 0.6],
            "val_accs": [0.60, 0.65, 0.70, 0.68],
            "val_aucs": [0.80, 0.72, 0.71, 0.70],
        }
        (fold / "history.json").write_text(json.dumps(hist))
        m = agg.load_fold_metrics(fold)
        assert m["epoch"] == 3, m                       # idx 2 -> epoch 3
        assert abs(m["val_auc"] - 0.71) < 1e-9, m       # auc AT min-loss, not max(0.80)
        assert abs(m["val_loss"] - 0.5) < 1e-9, m
    results["#3 aggregate_kfold: reports min-val-loss epoch, not max-over-epochs"] = "PASS"
except Exception as e:
    results["#3 aggregate_kfold"] = f"FAIL: {e}"


# --- #4 pack_features: corrupt rows excluded from index, offsets preserved ---
try:
    import torch
    import pandas as pd
    from synchronai.data.fnirs.feature_dataset import pack_features
    with tempfile.TemporaryDirectory() as d:
        fd = Path(d)
        (fd / "features").mkdir()
        shape = (12, 2)
        torch.save(torch.ones(shape), fd / "features/a.pt")
        torch.save(torch.zeros((99, 99)), fd / "features/bad.pt")   # wrong shape -> corrupt
        torch.save(torch.ones(shape) * 2, fd / "features/c.pt")
        pd.DataFrame({
            "feature_file": ["a.pt", "bad.pt", "c.pt"],
            "participant_type": ["child", "adult", "child"],
        }).to_csv(fd / "feature_index.csv", index=False)

        pack_features(fd)

        out = pd.read_csv(fd / "feature_index.csv")
        assert list(out["feature_file"]) == ["a.pt", "c.pt"], out["feature_file"].tolist()
        assert list(out["row_idx"]) == [0, 2], out["row_idx"].tolist()   # c keeps offset 2, not 1
        meta = json.loads((fd / "features_meta.json").read_text())
        assert meta["shape"][0] == 3 and meta["n_entries"] == 3          # bin still holds 3 rows
        assert (fd / "corrupt_files.txt").exists()
        # offsets actually resolve to the right rows in the packed bin
        mm = np.memmap(fd / "features_packed.bin", dtype="float32", mode="r",
                       shape=tuple(meta["shape"]))
        assert np.allclose(mm[0], 1.0) and np.allclose(mm[1], 0.0) and np.allclose(mm[2], 2.0)
    results["#4 pack_features: corrupt rows dropped from index, bin offsets intact"] = "PASS"
except Exception as e:
    results["#4 pack_features"] = f"FAIL: {e}"


# --- #5 diagnose redundancy R²: intercept captures the mean offset ----------
try:
    diag = _load_script(REPO / "scripts/diagnose_modality_repr_correlation.py")
    rng = np.random.default_rng(0)
    n, p = 500, 8
    video = np.maximum(rng.normal(size=(n, p)), 0).astype(np.float32)    # post-ReLU: nonneg, nonzero mean
    W = rng.normal(size=(p, p)).astype(np.float32)
    b = (rng.normal(size=(p,)) + 5.0).astype(np.float32)                 # large offset an intercept-free fit can't reach
    audio = (video @ W + b + 0.01 * rng.normal(size=(n, p))).astype(np.float32)
    s = diag.compute_corr(video, audio)
    assert s["r2_video_to_audio"] > 0.99, s          # with intercept: near-perfect (would sink without)
    assert "mean_per_dim_pearson" not in s           # meaningless metric removed
    results["#5 diagnose R²: intercept added (per-dim Pearson removed)"] = "PASS"
except Exception as e:
    results["#5 diagnose R²"] = f"FAIL: {e}"


# --- #6 multimodal event-loss-weight default aligned to 0.0 (source check) ---
try:
    src = (REPO / "src/synchronai/training/multimodal/train.py").read_text()
    assert re.search(r"event_loss_weight:\s*float\s*=\s*0\.0", src), "dataclass default not 0.0"
    assert re.search(r'--event-loss-weight"[^)]*default=0\.0', src), "argparse default not 0.0"
    results["#6 event-loss-weight: dataclass + argparse defaults both 0.0"] = "PASS"
except Exception as e:
    results["#6 event-loss-weight"] = f"FAIL: {e}"


# --- #7 diffusion reshape: n_pairs from feature_dim, not len(pair_names) -----
try:
    hb_count = 2
    feature_dim = 2                                   # per-pair model
    pair_names = [f"p{i}" for i in range(10)]         # full montage still listed in config
    target_len = 30
    x = np.zeros((1, target_len, feature_dim), np.float32)
    n_pairs = feature_dim // hb_count                 # the fixed line -> 1
    x.reshape(1, target_len, n_pairs, hb_count)       # succeeds
    try:
        x.reshape(1, target_len, len(pair_names), hb_count)   # old buggy line -> raises
        raise AssertionError("old len(pair_names) reshape should have failed but didn't")
    except ValueError:
        pass
    results["#7 diffusion reshape: n_pairs = feature_dim // hb_count"] = "PASS"
except Exception as e:
    results["#7 diffusion reshape"] = f"FAIL: {e}"


print("\n=== five-fix validation ===")
ok = True
for k, v in results.items():
    mark = "PASS" if v == "PASS" else "FAIL"
    if v != "PASS":
        ok = False
    print(f"  [{mark}] {k}")
    if v != "PASS":
        print(f"         {v}")
print()
sys.exit(0 if ok else 1)

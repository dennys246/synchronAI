"""Extract engineered hemodynamic features per per-pair fNIRS window.

Reviewer #4 control (ii): a hand-crafted-feature + logistic-regression baseline,
to test whether the pretrained NIRSgen encoder beats simple engineered features
(and thus whether 'the model learned hemodynamics' is supported beyond an internal
random-init ablation).

Mirrors scripts/extract_fnirs_features.py: same recordings, same per-pair windowing
(load_and_normalize_recording + window_recording), same normalization (encoder's
feature_mean/std from the diffusion config), same QC tiers, same feature_index.csv
schema + packed format. The ONLY difference is that each (472,2) HbO/HbR window is
reduced to a fixed engineered-feature vector instead of an encoder embedding — so it
plugs straight into the subject split and subject-level eval used everywhere else.

Per window, per HbO/HbR channel: mean, std, slope, range, peak |amplitude|,
time-to-peak, relative band power (task/Mayer/respiration), Hjorth activity/
mobility/complexity; plus HbO-HbR correlation and peak cross-correlation lag.

Output: feature_index.csv (+ row_idx) + features_packed.bin + features_meta.json.

Train the baseline on the result with scripts/fnirs_engineered_baseline.py.
Cluster-only (the fNIRS loader needs MNE); the feature math is numpy-only and unit
tested via _self_test().
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default physiological bands (Hz), matching the band-attribution analysis.
BANDS = [("task", 0.01, 0.05), ("mayer", 0.08, 0.12), ("respiration", 0.2, 0.4)]


def _chan_features(x, t, sfreq, bands):
    """Engineered features for one (n,) channel time series. Returns (names, vals)."""
    names, vals = [], []
    n = len(x)
    var = float(np.var(x))

    names += ["mean", "std", "slope", "range", "peak_abs", "ttp_frac"]
    mean = float(x.mean())
    std = float(x.std())
    # least-squares slope without polyfit's overhead; t is centered seconds
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    slope = float(np.dot(tc, x - mean) / denom) if denom > 0 else 0.0
    rng = float(x.max() - x.min())
    peak_abs = float(np.abs(x).max())
    ttp_frac = float(np.argmax(np.abs(x)) / max(n - 1, 1))  # 0..1 within window
    vals += [mean, std, slope, rng, peak_abs, ttp_frac]

    # Relative band power via rFFT (scale-robust; sums to ~1 across the spectrum)
    fft = np.fft.rfft(x - mean)
    psd = (fft.real ** 2 + fft.imag ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    total = float(psd.sum()) + 1e-12
    for bname, lo, hi in bands:
        names.append("bp_" + bname)
        mask = (freqs >= lo) & (freqs < hi)
        vals.append(float(psd[mask].sum()) / total)

    # Hjorth parameters
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_d = float(np.var(dx))
    var_dd = float(np.var(ddx))
    mobility = float(np.sqrt(var_d / var)) if var > 0 else 0.0
    mob_d = float(np.sqrt(var_dd / var_d)) if var_d > 0 else 0.0
    complexity = float(mob_d / mobility) if mobility > 0 else 0.0
    names += ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
    vals += [var, mobility, complexity]
    return names, vals


def compute_engineered_features(window, sfreq, bands=BANDS):
    """Engineered feature vector for one (n, 2) HbO/HbR window.

    window: real-signal samples only (drop any zero padding before calling).
    Returns (names, values) with values a float32 1-D array.
    """
    window = np.asarray(window, dtype=np.float64)
    n = window.shape[0]
    t = np.arange(n) / sfreq
    hbo, hbr = window[:, 0], window[:, 1]

    names, vals = [], []
    for cname, x in (("hbo", hbo), ("hbr", hbr)):
        cn, cv = _chan_features(x, t, sfreq, bands)
        names += [cname + "_" + k for k in cn]
        vals += cv

    # Cross-channel: Pearson correlation + peak cross-correlation lag (seconds)
    names += ["hbo_hbr_corr", "hbo_hbr_xcorr_lag_s"]
    if np.std(hbo) > 0 and np.std(hbr) > 0:
        corr = float(np.corrcoef(hbo, hbr)[0, 1])
        a = (hbo - hbo.mean()) / (np.std(hbo) * n)
        b = (hbr - hbr.mean()) / np.std(hbr)
        xc = np.correlate(a, b, mode="full")
        lag = float((np.argmax(xc) - (n - 1)) / sfreq)
    else:
        corr, lag = 0.0, 0.0
    vals += [corr, lag]

    vals = np.asarray(vals, dtype=np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return names, vals


def _self_test():
    """Numpy-only sanity checks for the feature math."""
    sf = 7.8125
    n = 469
    t = np.arange(n) / sf
    rng = np.random.default_rng(0)
    # a clean 0.1 Hz (Mayer-band) sine in HbO, noise in HbR
    hbo = np.sin(2 * np.pi * 0.1 * t)
    hbr = 0.01 * rng.standard_normal(n)
    names, vals = compute_engineered_features(np.stack([hbo, hbr], axis=1), sf)
    d = dict(zip(names, vals))
    assert abs(d["hbo_bp_mayer"]) > 0.8, ("Mayer power should dominate", d["hbo_bp_mayer"])
    assert d["hbo_bp_task"] < 0.1 and d["hbo_bp_respiration"] < 0.1
    # slope is in units/second: a 0->10 ramp over ~60 s gives ~0.167/s
    ramp = np.linspace(0, 10, n)
    _, v2 = compute_engineered_features(np.stack([ramp, ramp], axis=1), sf)
    slope = dict(zip(names, v2))["hbo_slope"]
    assert abs(slope - 10.0 / ((n - 1) / sf)) < 1e-3, ("slope units/sec", slope)
    # finite, fixed-length
    assert len(vals) == len(names) and np.isfinite(vals).all()
    print("OK: %d features/window:" % len(names))
    print("   ", names)
    print("    example Mayer-sine HbO band powers: task=%.3f mayer=%.3f resp=%.3f"
          % (d["hbo_bp_task"], d["hbo_bp_mayer"], d["hbo_bp_respiration"]))


def extract(args):
    import torch  # noqa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_fnirs_features import (
        discover_fnirs_paths, extract_subject_id, detect_participant_type,
        load_and_normalize_recording, window_recording,
    )
    import pandas as pd

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = BANDS

    # Encoder .pt is loaded ONLY for its config (norm stats + pair/window params);
    # the encoder itself is never instantiated for engineered features.
    save_data = torch.load(args.encoder_weights, map_location="cpu", weights_only=False)
    cfg = save_data["config"]
    feature_mean = np.array(cfg["feature_mean"], dtype=np.float32)
    feature_std = np.array(cfg["feature_std"], dtype=np.float32)
    target_pairs = cfg["pair_names"]
    hb_types = cfg["hb_types"]
    target_sfreq = cfg["sfreq_hz"]
    model_len = save_data["encoder_config"]["input_length"]
    real_len = int(cfg.get("target_len", model_len))
    logger.info("config: sfreq=%.4f model_len=%d real_len=%d", target_sfreq, model_len, real_len)

    qc_tier = {}
    if args.qc_cache and Path(args.qc_cache).exists():
        qc_df = pd.read_csv(args.qc_cache)
        for _, row in qc_df.iterrows():
            qc_tier[row["fnirs_path"]] = row["quality_tier"]
        logger.info("loaded %d QC tiers from %s", len(qc_tier), args.qc_cache)

    paths = discover_fnirs_paths(args.data_dirs, "hemodynamic")
    logger.info("discovered %d recordings", len(paths))

    feat_names = None
    rows = []          # feature_index.csv rows
    feat_buf = []      # list of (feature_dim,) float32 vectors, packed order
    n_ok = n_fail = 0
    t0 = time.time()

    for i, fnirs_path in enumerate(paths):
        subject_id = extract_subject_id(fnirs_path)
        participant_type = detect_participant_type(fnirs_path)
        tier = qc_tier.get(fnirs_path, "unknown")
        if args.include_tiers and tier not in args.include_tiers:
            continue
        try:
            pairs = load_and_normalize_recording(
                fnirs_path, feature_mean=feature_mean, feature_std=feature_std,
                target_pairs=target_pairs, hb_types=hb_types,
                target_sfreq=target_sfreq, per_pair=True,
            )
        except Exception as e:  # noqa
            logger.warning("load failed %s: %s", fnirs_path, e)
            pairs = None
        if not pairs:
            n_fail += 1
            continue

        for pair_name, pair_arr in pairs:
            windows = window_recording(
                pair_arr, model_len=model_len, stride_seconds=60.0,
                sfreq_hz=target_sfreq, real_len=real_len,
            )
            for seg_idx, w in enumerate(windows):
                names, vals = compute_engineered_features(w[:real_len], target_sfreq, bands)
                if feat_names is None:
                    feat_names = names
                feat_buf.append(vals)
                rows.append({
                    "feature_file": "engineered",  # packed-only; no per-window .pt
                    "fnirs_path": fnirs_path,
                    "subject_id": subject_id,
                    "participant_type": participant_type,
                    "window_idx": seg_idx,
                    "pair_name": pair_name,
                    "feature_dim": len(vals),
                    "n_frames": 1,
                    "multiscale": False,
                    "quality_tier": tier,
                })
        n_ok += 1
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d recordings, %d windows, %.0fs",
                        i + 1, len(paths), len(feat_buf), time.time() - t0)
        if args.limit and n_ok >= args.limit:
            logger.info("stopping at --limit %d recordings", args.limit)
            break

    if not feat_buf:
        raise SystemExit("no engineered features produced")

    feats = np.stack(feat_buf, axis=0).astype(np.float32)
    n, fdim = feats.shape
    # write packed bin + meta + index (row_idx == packed order)
    feats.tofile(out_dir / "features_packed.bin")
    with open(out_dir / "features_meta.json", "w") as f:
        json.dump({"shape": [n, fdim], "dtype": "float32", "n_entries": n,
                   "per_entry_shape": [fdim], "n_corrupt": 0}, f, indent=2)
    df = pd.DataFrame(rows)
    df["row_idx"] = range(n)
    df.to_csv(out_dir / "feature_index.csv", index=False)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feat_names, f, indent=2)
    logger.info("wrote %d windows x %d engineered features to %s (ok=%d fail=%d, %.0fs)",
                n, fdim, out_dir, n_ok, n_fail, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run numpy-only feature-math checks and exit")
    ap.add_argument("--data-dirs", help="comma-separated fNIRS study dirs")
    ap.add_argument("--encoder-weights",
                    help="fnirs_unet_encoder.pt — loaded ONLY for norm/window config")
    ap.add_argument("--output-dir")
    ap.add_argument("--qc-cache", default=None, help="data/qc_tiers.csv")
    ap.add_argument("--include-tiers", default="gold,standard",
                    help="comma-separated tiers to keep (match the classifier)")
    ap.add_argument("--limit", type=int, default=0, help="cap #recordings (smoke test)")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    for req in ("data_dirs", "encoder_weights", "output_dir"):
        if not getattr(args, req):
            ap.error("--%s is required (unless --self-test)" % req.replace("_", "-"))
    args.include_tiers = [t.strip() for t in args.include_tiers.split(",") if t.strip()] \
        if args.include_tiers else None
    extract(args)


if __name__ == "__main__":
    main()

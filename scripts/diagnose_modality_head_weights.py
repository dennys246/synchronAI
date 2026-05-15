#!/usr/bin/env python3
"""Inspect a multimodal classifier's head weights to detect modality bias.

Tests Diagnostic 1 from docs/audio_contribution_investigation.md.

For a v2/v3 multimodal classifier, the head's first Linear layer takes
concat[video_repr (P-dim), audio_repr (P-dim)] as input. Split that weight
matrix into video and audio columns; the relative magnitudes tell us
whether the head is using both modalities or has collapsed to one.

  Ratio (||W_video|| / ||W_audio||) near 1.0  →  modalities used roughly equally
  Ratio >> 1 (e.g., > 5)                       →  head functionally ignores audio
  Ratio << 1                                    →  head ignores video (unexpected)

Usage (single checkpoint):
    python scripts/diagnose_modality_head_weights.py \\
        --checkpoint runs/multimodal_features/v2_baseline_v6/best.pt

Usage (CV directory — aggregates over folds):
    python scripts/diagnose_modality_head_weights.py \\
        --cv-dir runs/multimodal_features/v2_baseline_v6_cv5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def inspect_checkpoint(ckpt_path: Path) -> dict:
    """Load a checkpoint and return head weight diagnostics.

    The head is `nn.Sequential(Linear(2P, head_hidden), GELU, Dropout, Linear(head_hidden, 1))`.
    For V2 and V3, the first Linear's input is concat[video_repr, audio_repr]. Split.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    state = ckpt["model_state_dict"]

    arch = config.get("arch", "v2")
    proj_dim = config.get("video_hidden", 64)  # V2/V3 use video_hidden as proj_dim

    # The first Linear of the head is "head.0.weight" (Sequential indexing).
    # For V1 (legacy), the same path applies — its head also starts with a Linear.
    w_key = "head.0.weight"
    if w_key not in state:
        raise RuntimeError(
            f"{ckpt_path}: 'head.0.weight' not found in state_dict. "
            f"Arch was {arch}; available head-like keys: "
            f"{[k for k in state if 'head' in k][:5]}"
        )

    W = state[w_key].cpu().numpy()  # shape (head_hidden, 2P) for V2/V3
    if W.shape[1] != 2 * proj_dim:
        raise RuntimeError(
            f"{ckpt_path}: expected head.0 input dim 2*proj_dim={2*proj_dim} "
            f"(arch={arch}, proj_dim={proj_dim}), got {W.shape[1]}. "
            f"This diagnostic only handles V2/V3 concat-fusion heads."
        )

    W_video = W[:, :proj_dim]
    W_audio = W[:, proj_dim:]

    # Frobenius norm of the whole block: total "weight mass" allocated to each modality.
    norm_video = float(np.linalg.norm(W_video, ord="fro"))
    norm_audio = float(np.linalg.norm(W_audio, ord="fro"))
    ratio = norm_video / max(norm_audio, 1e-12)

    # Mean of absolute weights — robust to outliers and easier to interpret per-unit.
    mean_abs_video = float(np.abs(W_video).mean())
    mean_abs_audio = float(np.abs(W_audio).mean())

    # Per-output-unit ratios (one ratio per head_hidden neuron). Distribution shows
    # whether ALL output units ignore audio or only some.
    per_unit_video = np.linalg.norm(W_video, axis=1)  # (head_hidden,)
    per_unit_audio = np.linalg.norm(W_audio, axis=1)
    per_unit_ratio = per_unit_video / np.maximum(per_unit_audio, 1e-12)

    return {
        "arch": arch,
        "proj_dim": proj_dim,
        "head_hidden": int(W.shape[0]),
        "norm_video_fro": norm_video,
        "norm_audio_fro": norm_audio,
        "fro_ratio_video_over_audio": ratio,
        "mean_abs_video": mean_abs_video,
        "mean_abs_audio": mean_abs_audio,
        "per_unit_ratio_median": float(np.median(per_unit_ratio)),
        "per_unit_ratio_min": float(per_unit_ratio.min()),
        "per_unit_ratio_max": float(per_unit_ratio.max()),
    }


def print_one(name: str, d: dict) -> None:
    print(f"  {name:<20}  arch={d['arch']:<3} P={d['proj_dim']:<3} H={d['head_hidden']:<3}  "
          f"||V||={d['norm_video_fro']:>6.3f}  ||A||={d['norm_audio_fro']:>6.3f}  "
          f"V/A ratio = {d['fro_ratio_video_over_audio']:>6.3f}  "
          f"per-unit V/A median={d['per_unit_ratio_median']:.3f} (range {d['per_unit_ratio_min']:.3f}-{d['per_unit_ratio_max']:.3f})")


def interpret(ratio: float) -> str:
    if 0.5 <= ratio <= 2.0:
        return "balanced — head uses both modalities roughly equally"
    if ratio > 5.0:
        return "STRONG video bias — head functionally ignores audio (H1 likely)"
    if ratio > 2.0:
        return "video-leaning — audio used but de-weighted"
    if ratio < 0.2:
        return "STRONG audio bias — head ignores video (unexpected!)"
    return "audio-leaning — video used but de-weighted (unexpected!)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, help="Single best.pt to inspect")
    g.add_argument("--cv-dir", type=Path, help="CV directory containing fold_*/best.pt")
    parser.add_argument(
        "--checkpoint-name", default="best.pt",
        help="Which checkpoint inside each fold dir (default best.pt; also useful: best_acc.pt)",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        d = inspect_checkpoint(args.checkpoint)
        print(f"=== {args.checkpoint} ===")
        print_one(args.checkpoint.parent.name, d)
        print()
        print(f"Interpretation: {interpret(d['fro_ratio_video_over_audio'])}")
        return

    cv_dir = args.cv_dir
    fold_dirs = sorted(p for p in cv_dir.glob("fold_*") if (p / args.checkpoint_name).exists())
    if not fold_dirs:
        raise SystemExit(f"No fold_*/{args.checkpoint_name} under {cv_dir}")

    print(f"=== {cv_dir} ({len(fold_dirs)} folds) ===")
    print()
    ratios = []
    for fd in fold_dirs:
        d = inspect_checkpoint(fd / args.checkpoint_name)
        print_one(fd.name, d)
        ratios.append(d["fro_ratio_video_over_audio"])

    ratios = np.array(ratios)
    print()
    print(f"=== Summary across {len(ratios)} folds ===")
    print(f"  Frobenius ratio V/A: mean={ratios.mean():.3f}  std={ratios.std():.3f}  "
          f"range=[{ratios.min():.3f}, {ratios.max():.3f}]")
    print(f"  Interpretation: {interpret(float(ratios.mean()))}")


if __name__ == "__main__":
    main()

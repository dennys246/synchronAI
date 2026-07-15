#!/usr/bin/env python3
"""D1b: measure correlation between video_repr and audio_repr on val data.

Tests whether the multimodal model's per-modality representations are
redundant (encode the same info) vs orthogonal (encode different info that
the model isn't combining productively).

Loads a trained checkpoint, hooks the video_repr / audio_repr layers,
runs forward on val data (from the SAME fold the checkpoint was trained on),
and reports four similarity measures:

  - Mean cosine similarity (per-sample)
  - Mean per-dim Pearson correlation (matched dims)
  - Linear R² (video_repr → audio_repr)
  - Linear R² (audio_repr → video_repr)

The two linear R² values are the most informative: they answer "can one
modality's representation be linearly predicted from the other's?"

  - R² > 0.7: representations are strongly redundant
  - 0.3 < R² < 0.7: moderate overlap
  - R² < 0.3: representations are largely independent

A high redundancy + zero multimodal benefit is consistent with H2/H4:
the audio features encode info also captured by video. Different audio
features (per-layer WavLM, emotion2vec) might break that redundancy.

A low redundancy + zero multimodal benefit is more puzzling: the model
has two independent views but doesn't benefit. Points to H4 (audio features
just don't carry synchrony-discriminative signal even though they're
distinct from video).

Usage:
    python scripts/diagnose_modality_repr_correlation.py \\
        --checkpoint runs/multimodal_features/v2_baseline_v6_cv5/fold_0/best.pt

    # Aggregate over folds:
    python scripts/diagnose_modality_repr_correlation.py \\
        --cv-dir runs/multimodal_features/v2_baseline_v6_cv5
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _load_train_module():
    """Import scripts/train_multimodal_from_features.py without side effects."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "train_mm", here / "train_multimodal_from_features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Make the synchronai package importable for the embedded imports.
    repo = here.parent
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec.loader.exec_module(mod)
    return mod


def build_model(arch: str, config: dict, video_dim: int, audio_dim: int, train_mm) -> torch.nn.Module:
    """Reconstruct the model class matching the training-time arch."""
    proj_dim = config["video_hidden"]
    head_hidden = config["head_hidden"]
    dropout = config["dropout"]
    if arch == "v2":
        return train_mm.MultiModalV2(
            video_feature_dim=video_dim, audio_feature_dim=audio_dim,
            proj_dim=proj_dim, head_hidden=head_hidden,
            proj_dropout=dropout, lstm_dropout=0.2,
            repr_dropout=dropout, head_dropout=dropout,
        )
    if arch == "v3":
        return train_mm.MultiModalV3(
            video_feature_dim=video_dim, audio_feature_dim=audio_dim,
            proj_dim=proj_dim, head_hidden=head_hidden,
            proj_dropout=dropout, lstm_dropout=0.2,
            repr_dropout=dropout, head_dropout=dropout,
            attn_heads=4, attn_dropout=0.2,
        )
    raise ValueError(f"D1b only supports v2/v3 (concat-head architecture). Got arch={arch!r}.")


def compute_corr(video_repr: np.ndarray, audio_repr: np.ndarray) -> dict:
    """Compute correlation measures between video and audio representations.

    video_repr, audio_repr: shape (N, P)
    """
    n, p = video_repr.shape
    assert audio_repr.shape == (n, p)

    # 1. Per-sample cosine similarity, averaged.
    v_norm = np.linalg.norm(video_repr, axis=1) + 1e-12
    a_norm = np.linalg.norm(audio_repr, axis=1) + 1e-12
    cos_sim = (video_repr * audio_repr).sum(axis=1) / (v_norm * a_norm)
    mean_cos = float(cos_sim.mean())

    # 2/3. Linear R² with an intercept, both directions. The intercept column is
    # essential: the reprs are post-ReLU (non-zero-mean), so an intercept-free
    # lstsq cannot fit the mean and biases R² LOW — which can flip interpret() to
    # a spurious "LOW redundancy → H4". (A per-dim Pearson between dim i of video
    # and dim i of audio was intentionally dropped: the two projection heads share
    # no per-dimension correspondence, so it measured nothing.)
    ones = np.ones((n, 1), dtype=video_repr.dtype)

    def _r2(x_repr, y_repr):
        design = np.concatenate([x_repr, ones], axis=1)
        w, *_ = np.linalg.lstsq(design, y_repr, rcond=None)
        y_pred = design @ w
        ss_res = ((y_repr - y_pred) ** 2).sum()
        ss_tot = ((y_repr - y_repr.mean(0, keepdims=True)) ** 2).sum()
        return float(1 - ss_res / max(ss_tot, 1e-12))

    r2_v_to_a = _r2(video_repr, audio_repr)
    r2_a_to_v = _r2(audio_repr, video_repr)

    return {
        "n_val_samples": n,
        "proj_dim": p,
        "mean_cosine_sim": mean_cos,
        "r2_video_to_audio": r2_v_to_a,
        "r2_audio_to_video": r2_a_to_v,
    }


def inspect_checkpoint(ckpt_path: Path) -> dict:
    """Load checkpoint, extract video_repr + audio_repr on val data, compute correlations."""
    train_mm = _load_train_module()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    arch = config["arch"]

    # Reconstruct val split exactly as during training.
    video_dir = Path(config["video_feature_dir"])
    audio_dir = Path(config["audio_feature_dir"])
    merged, video_dim, _, audio_dim, _ = train_mm.merge_feature_indices(video_dir, audio_dir)
    entries = merged.to_dict("records")
    num_folds = config.get("num_folds", 0)
    fold_idx = config.get("fold_idx", -1)
    # Reconstruct the SAME split the run trained with. Runs default to family_id
    # grouping now; using the run's recorded key avoids analyzing a different
    # (subject-level, leaky) val set than training used.
    group_key = config.get("group_key", "subject_id")
    if num_folds and num_folds > 0 and fold_idx is not None and fold_idx >= 0:
        _, val_entries = train_mm.subject_kfold_split(
            entries, num_folds, fold_idx, config["seed"], group_key=group_key)
    else:
        _, val_entries = train_mm.subject_grouped_split(
            entries, config["val_split"], config["seed"], group_key=group_key)

    # Build model and load weights.
    model = build_model(arch, config, video_dim, audio_dim, train_mm)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Hook the per-modality repr outputs. Both V2 and V3 have
    # video_repr_drop and audio_repr_drop as the final per-modality layers.
    video_captures: list = []
    audio_captures: list = []

    def video_hook(_module, _inp, out):
        video_captures.append(out.detach().cpu().numpy())

    def audio_hook(_module, _inp, out):
        audio_captures.append(out.detach().cpu().numpy())

    h1 = model.video_repr_drop.register_forward_hook(video_hook)
    h2 = model.audio_repr_drop.register_forward_hook(audio_hook)

    # Preload val data (same dataset class used in training).
    dataset = train_mm.MultiModalFeatureDataset(video_dir, audio_dir, val_entries)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=train_mm.collate)

    with torch.inference_mode():
        for batch in loader:
            _ = model(batch["video_features"], batch["audio_features"])

    h1.remove()
    h2.remove()

    video_repr = np.concatenate(video_captures, axis=0)
    audio_repr = np.concatenate(audio_captures, axis=0)

    stats = compute_corr(video_repr, audio_repr)
    stats["arch"] = arch
    stats["fold_idx"] = fold_idx
    return stats


def print_one(name: str, s: dict) -> None:
    print(f"  {name:<14}  arch={s['arch']:<3} P={s['proj_dim']:<3} N_val={s['n_val_samples']:<5}  "
          f"cos={s['mean_cosine_sim']:>+.4f}  "
          f"R²(V→A)={s['r2_video_to_audio']:>+.4f}  R²(A→V)={s['r2_audio_to_video']:>+.4f}")


def interpret(r2_v_to_a: float, r2_a_to_v: float) -> str:
    r2_max = max(r2_v_to_a, r2_a_to_v)
    if r2_max > 0.7:
        return ("STRONG redundancy — modalities encode overlapping info. "
                "H2/H4 likely; new audio rep may break redundancy.")
    if r2_max > 0.3:
        return "Moderate overlap — modalities partially redundant."
    return ("LOW redundancy — representations are largely independent. "
            "H4 likely (WavLM features just don't add discriminative signal).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, help="Single best.pt to inspect")
    g.add_argument("--cv-dir", type=Path, help="CV directory containing fold_*/best.pt")
    parser.add_argument(
        "--checkpoint-name", default="best.pt",
        help="Which checkpoint inside each fold dir (default best.pt)",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        s = inspect_checkpoint(args.checkpoint)
        print(f"=== {args.checkpoint} ===")
        print_one(args.checkpoint.parent.name, s)
        print()
        print(f"Interpretation: {interpret(s['r2_video_to_audio'], s['r2_audio_to_video'])}")
        return

    cv_dir = args.cv_dir
    fold_dirs = sorted(p for p in cv_dir.glob("fold_*") if (p / args.checkpoint_name).exists())
    if not fold_dirs:
        raise SystemExit(f"No fold_*/{args.checkpoint_name} under {cv_dir}")

    print(f"=== {cv_dir} ({len(fold_dirs)} folds) ===\n")
    all_stats = []
    for fd in fold_dirs:
        s = inspect_checkpoint(fd / args.checkpoint_name)
        print_one(fd.name, s)
        all_stats.append(s)

    r2_v_to_a = np.array([s["r2_video_to_audio"] for s in all_stats])
    r2_a_to_v = np.array([s["r2_audio_to_video"] for s in all_stats])
    cos = np.array([s["mean_cosine_sim"] for s in all_stats])
    print()
    print(f"=== Summary across {len(all_stats)} folds ===")
    print(f"  Mean cosine similarity: {cos.mean():+.4f} ± {cos.std():.4f}  range=[{cos.min():+.4f}, {cos.max():+.4f}]")
    print(f"  R²(V→A):                {r2_v_to_a.mean():+.4f} ± {r2_v_to_a.std():.4f}  range=[{r2_v_to_a.min():+.4f}, {r2_v_to_a.max():+.4f}]")
    print(f"  R²(A→V):                {r2_a_to_v.mean():+.4f} ± {r2_a_to_v.std():.4f}  range=[{r2_a_to_v.min():+.4f}, {r2_a_to_v.max():+.4f}]")
    print()
    print(f"Interpretation: {interpret(float(r2_v_to_a.mean()), float(r2_a_to_v.mean()))}")


if __name__ == "__main__":
    main()

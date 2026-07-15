"""Score a trained multimodal checkpoint on a held-out feature set (cold).

Purpose: the cross-study number for the EmoGrow question -- train on
CARE + P-CAT (+R56) with NO EmoGrow subjects, then score EmoGrow features here.
`train_multimodal_from_features.py` only does internal k-fold subject CV; it has
no eval-only path, so this loads a saved checkpoint and runs one forward pass
over an external feature dir.

Reuses training's exact code (imported, not reimplemented): merge_feature_indices,
MultiModalFeatureDataset, collate, compute_metrics, and the model classes. The
model is rebuilt from the run's config.json so arch/width match the checkpoint.

Supports the standard-dataset arches v1-v4 (the ceiling lineup). The patch/pose/
crossperson/context arches need extra feature dirs EmoGrow doesn't have and are
out of scope here.

Run on the cluster (needs ml-env torch + the feature dirs). Example:
  ML_PY=$SYNCHRONAI_DIR/ml-env/bin/python
  $ML_PY scripts/eval_multimodal_holdout.py \
      --run-dir runs/multimodal_features/lstm_concat \
      --video-feature-dir data/dinov2_features_meanpatch_emogrow \
      --audio-feature-dir data/wavlm_baseplus_features_emogrow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_multimodal_from_features import (  # noqa: E402
    MultiModalFeatureDataset,
    MultiModalLSTMConcat,
    MultiModalV2,
    MultiModalV3,
    MultiModalV4,
    collate,
    compute_metrics,
    merge_feature_indices,
)


def build_model(arch: str, cfg: dict, video_dim: int, audio_dim: int,
                video_frames: int, audio_frames: int, device: str):
    """Mirror the arch->model construction in train_multimodal_from_features.main."""
    vh = cfg["video_hidden"]
    ah = cfg["audio_hidden"]
    hh = cfg["head_hidden"]
    do = cfg["dropout"]
    if arch == "v1":
        return MultiModalLSTMConcat(
            video_feature_dim=video_dim, audio_feature_dim=audio_dim,
            video_hidden=vh, audio_hidden=ah, head_hidden=hh, dropout=do,
        ).to(device)
    if arch == "v2":
        return MultiModalV2(
            video_feature_dim=video_dim, audio_feature_dim=audio_dim,
            proj_dim=vh, head_hidden=hh,
            proj_dropout=do, lstm_dropout=0.2, repr_dropout=do, head_dropout=do,
        ).to(device)
    if arch == "v3":
        return MultiModalV3(
            video_feature_dim=video_dim, audio_feature_dim=audio_dim,
            proj_dim=vh, head_hidden=hh,
            proj_dropout=do, lstm_dropout=0.2, repr_dropout=do, head_dropout=do,
            attn_heads=4, attn_dropout=0.2,
        ).to(device)
    if arch == "v4":
        return MultiModalV4(
            video_feature_dim=video_dim, audio_feature_dim=audio_dim,
            proj_dim=vh, head_hidden=hh,
            proj_dropout=do, head_dropout=do,
            n_layers=1, n_heads=4, attn_dropout=0.1,
            n_video_frames=video_frames, n_audio_frames=audio_frames,
        ).to(device)
    raise SystemExit(
        f"arch '{arch}' not supported by this eval script (v1-v4 only). "
        f"patch/pose/crossperson/context arches need extra feature dirs."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="trained run dir (has config.json + checkpoint)")
    ap.add_argument("--video-feature-dir", required=True)
    ap.add_argument("--audio-feature-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt",
                    help="checkpoint filename inside run-dir (best.pt|best_auc.pt|...)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    arch = cfg["arch"]
    modality = cfg.get("modality", "multimodal")

    video_dir = Path(args.video_feature_dir)
    audio_dir = Path(args.audio_feature_dir)
    merged, video_dim, video_frames, audio_dim, audio_frames = merge_feature_indices(
        video_dir, audio_dir, pose_feature_dir=None,
    )
    # Dims must match what the checkpoint was trained on, else the load is silently wrong.
    for name, got, exp in [("video_dim", video_dim, cfg.get("video_dim")),
                           ("audio_dim", audio_dim, cfg.get("audio_dim"))]:
        if exp is not None and got != exp:
            raise SystemExit(f"{name} mismatch: eval features {got} vs checkpoint {exp}")

    entries = merged.to_dict("records")
    dataset = MultiModalFeatureDataset(video_dir, audio_dir, entries, audio_shuffle_prob=0.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    model = build_model(arch, cfg, video_dim, audio_dim, video_frames, audio_frames, device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"loaded {arch} from {run_dir / args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}), scoring {len(entries)} windows on {device}")

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            v = batch["video_features"].to(device)
            a = batch["audio_features"].to(device)
            y = batch["label"]
            if modality == "video":
                a = torch.zeros_like(a)
            elif modality == "audio":
                v = torch.zeros_like(v)
            logits = model(v, a)
            all_logits.append(logits.cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    m = compute_metrics(logits, labels)

    # Per-subject AUC too -- effective N is subjects, so the spread matters.
    subj = [e["subject_id"] for e in entries]
    print("\n==== held-out eval ====")
    print(f"  windows: {len(labels)}   subjects: {len(set(subj))}")
    print(f"  overall: AUC={m['auc']:.4f}  acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()

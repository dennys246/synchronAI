#!/bin/bash
SCRIPT_VERSION="submit_patch_probe_fold0-v2"
# v2: switched MM_VIDEO_FEATURE_DIR from data/dinov2_features (broken — that
# dir holds 2D CLS slices, see feedback_torch_save_view.md) to
# data/dinov2_features_patches (produced by dinov2_patch_extract_bsub.sh).
# Original v1 failed in the trainer's MultiModalPatchFeatureDataset with
# "expects 3D video features (T, P, D); got torch.Size([12, 768])".
# =============================================================================
# Probe 1: does patch-level DINOv2 (vs. spatial mean-pool) break the
# train_acc ~0.77 ceiling?
#
# The capacity probe (submit_capacity_probe_fold0) confirmed that train_acc
# saturates at ~0.77 across h=64/128/256 even unregularized. Combined with
# the regularization probe, that points to a feature ceiling rather than a
# capacity or regularization ceiling.
#
# But the v2 video pathway mean-pools 257 DINOv2 patch tokens per frame
# into a single 768-dim vector BEFORE the model sees the data. If the
# meaningful synchrony signal lives in *which patch* (e.g., the face, the
# hands), that information is destroyed at the input.
#
# This probe uses the full patch grid (12, 257, 768) per second instead
# of the mean-pooled (12, 768), with a learnable spatial-attention query
# per frame. Same v2 fusion + audio path; same fold 0 + seed 42; same
# low-reg + no-early-stop config as the prior probes for direct comparison.
#
# Decision rule (after 30 epochs):
#   - train_acc breaks past ~0.85 → spatial information matters,
#     meanpatch was the bottleneck. Run full CV with v2_patch and rewrite
#     the grant around "patch-level visual features + more subjects".
#   - train_acc still flat at ~0.77 → spatial info doesn't help here,
#     the ceiling is upstream of DINOv2 itself (label noise, second-
#     granularity, or signal isn't in vision at all). Grant needs to
#     argue for new pretrained backbones (probe 3) and/or new objective
#     (probe 2).
#
# Reference points (fold 0, seed=42):
#   v2_baseline_v6_cv5/fold_0     : meanpatch, dropout=0.3, wd=1e-2, train_max=0.746
#   v2_regprobe_fold0_lowdrop     : meanpatch, dropout=0.05, wd=0.0,  train_max=0.758
#   v2_caprobe_h256_fold0         : meanpatch, h=256 low-reg,         train_max=0.772
#   THIS RUN: v2_patch_probe_fold0 : patch-grid, low-reg
#
# WARNING: this run is slower than the meanpatch probes. Patch features
# are 9.5MB/file (vs 50KB for meanpatch) so the dataset is lazy-loaded
# from disk per __getitem__. NFS read latency + spatial-attention compute
# dominate. Expect 30+ min/epoch vs ~2 min/epoch for meanpatch. 30 epochs
# may take 12-20 hours wall clock.
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

export MM_ARCH="v2_patch"
export MM_VIDEO_FEATURE_DIR="data/dinov2_features_patches"
export MM_AUDIO_FEATURE_DIR="data/wavlm_baseplus_features"
export MM_SAVE_DIR="runs/multimodal_features/v2_patch_probe_fold0"
export MM_VIDEO_HIDDEN="64"
export MM_AUDIO_HIDDEN="64"
export MM_HEAD_HIDDEN="64"
export MM_DROPOUT="0.05"
export MM_WEIGHT_DECAY="0.0"
export MM_EPOCHS="30"
export MM_PATIENCE="30"
export MM_NUM_FOLDS="5"
export MM_FOLD_IDX="0"
export MM_EARLY_STOP_METRIC="val_loss"
# Lazy-load per-sample disk reads — give the loader a couple of workers
# so I/O overlaps with the spatial-attention compute.
export MM_NUM_WORKERS="2"

echo "=== [$SCRIPT_VERSION] Submitting patch-level DINOv2 probe (fold 0, low-reg) ==="
echo "  arch:       $MM_ARCH"
echo "  video:      $MM_VIDEO_FEATURE_DIR  (3D patch features, lazy-loaded)"
echo "  audio:      $MM_AUDIO_FEATURE_DIR"
echo "  save_dir:   $MM_SAVE_DIR"
echo "  dropout:    $MM_DROPOUT"
echo "  wd:         $MM_WEIGHT_DECAY"
echo "  epochs:     $MM_EPOCHS  patience: $MM_PATIENCE  (no early stop)"
echo "  fold:       $MM_FOLD_IDX / $MM_NUM_FOLDS"
echo "  num_workers: $MM_NUM_WORKERS"
echo "  Note: per-epoch wall time ~30 min (vs ~2 min meanpatch); total ~12-20h"

sh "$LAUNCHER"

echo ""
echo "=== Submitted. Check status: bjobs ==="
echo "Once complete, add this row to the capacity curve:"
echo "  bash scripts/bsub/capacity_probe_summary.sh"
echo "(Then manually note the v2_patch result vs the meanpatch ceiling.)"

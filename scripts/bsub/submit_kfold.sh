#!/bin/bash
SCRIPT_VERSION="submit_kfold-v1"
# =============================================================================
# Submit N-fold CV for a single multimodal config. Each fold becomes its own
# LSF job; the union of folds gives a deterministic, non-overlapping val
# partition across the 49 subjects.
#
# Why: prior architecture sweeps used fold-0-only single-fold val splits.
# v3_h24 5-fold CV revealed fold 0 was a +0.07 outlier above the next-best
# fold — meaning every "best of 0.76" number we'd been quoting was lucky
# fold composition, not generalization. To make defensible architecture
# claims at this dataset size, every comparison needs CV.
#
# Usage: set env vars to specify what to CV, then run:
#
#   MM_ARCH=v2 \
#   MM_VIDEO_HIDDEN=64 MM_AUDIO_HIDDEN=64 MM_HEAD_HIDDEN=64 \
#   MM_DROPOUT=0.3 MM_WEIGHT_DECAY=1e-2 \
#   MM_NUM_FOLDS=5 \
#   MM_SAVE_DIR_BASE=runs/multimodal_features/v2_baseline_v6_cv5 \
#     bash scripts/bsub/submit_kfold.sh
#
# Each fold writes to ${MM_SAVE_DIR_BASE}/fold_K/. After all folds finish:
#
#   python scripts/aggregate_kfold_results.py --cv-dir $MM_SAVE_DIR_BASE
#
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"
NUM_FOLDS="${MM_NUM_FOLDS:-5}"
SAVE_DIR_BASE="${MM_SAVE_DIR_BASE:?MM_SAVE_DIR_BASE must be set (e.g., runs/multimodal_features/v2_baseline_v6_cv5)}"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

echo "=== [$SCRIPT_VERSION] Submitting ${NUM_FOLDS}-fold CV ==="
echo "  arch:           ${MM_ARCH:-v1}"
echo "  hidden dims:    video=${MM_VIDEO_HIDDEN:-64} audio=${MM_AUDIO_HIDDEN:-64} head=${MM_HEAD_HIDDEN:-64}"
echo "  dropout / wd:   ${MM_DROPOUT:-0.3} / ${MM_WEIGHT_DECAY:-1e-2}"
echo "  save_dir_base:  $SAVE_DIR_BASE"
if [ -n "${MM_AUDIO_FEATURE_DIR:-}" ]; then
    echo "  audio features: $MM_AUDIO_FEATURE_DIR"
fi

submit_fold() {
    local fold="$1"
    local save_dir="${SAVE_DIR_BASE}/fold_${fold}"
    echo ""
    echo "--- Submitting fold $fold of $NUM_FOLDS → $save_dir ---"
    # Subshell so per-fold env vars don't leak. Invoke the launcher via `sh`
    # to dodge NFS exec-bit drops; matches the pattern in submit_v2_sweep.
    (
        export MM_SAVE_DIR="$save_dir"
        export MM_NUM_FOLDS="$NUM_FOLDS"
        export MM_FOLD_IDX="$fold"
        sh "$LAUNCHER"
    )
}

for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    submit_fold "$fold"
done

echo ""
echo "=== All $NUM_FOLDS folds submitted. Check status: bjobs ==="
echo "Once all complete:"
echo "  python scripts/aggregate_kfold_results.py --cv-dir $SAVE_DIR_BASE"

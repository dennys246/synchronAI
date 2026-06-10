#!/bin/bash
SCRIPT_VERSION="submit_capacity_probe_fold0-v1"
# =============================================================================
# Capacity probe: is the train_acc ~0.76 ceiling a feature ceiling or a
# capacity ceiling?
#
# Prior result (submit_regprobe_fold0): at h=64 with dropout=0.05 and wd=0,
# train_acc plateaus at 0.758 after ~12 epochs and stays flat for the rest
# of the run. That rules out regularization as the cause. It does NOT yet
# rule out "the projection bottleneck is too tight" — 768→64 deletes 92%
# of feature variance, and v2 LSTM(64→64) might just not have enough
# parameters to memorize 48K samples even unregularized.
#
# This probe re-runs the same low-reg setup at two larger capacities
# (h=128 and h=256, both with same dropout=0.05 wd=0 epochs=30 patience=30),
# fold 0 only, paired with the existing h=64 low-reg run to form a
# 3-point capacity curve under identical training conditions.
#
# Decision rule (after both finish):
#   - train_acc rises monotonically (e.g. 0.76 → 0.82 → 0.88) → capacity-
#     limited; bigger model finds more signal. Run full CV at the best
#     capacity to check if val tracks.
#   - train_acc flat across capacities (0.76 → 0.76 → 0.76) → feature-
#     limited; the DINOv2-meanpatch + WavLM-base-plus features genuinely
#     cap at ~76% accuracy on this task. Grant case needs to be about
#     different features, not more subjects.
#   - intermediate (some rise but val unchanged) → partial capacity gain
#     without generalization → likely overfitting onto subject identity.
#
# Reference points (fold 0, same seed=42, all from runs/multimodal_features):
#   h=64,  reg=default : v2_baseline_v6_cv5/fold_0      train_max=0.746
#   h=64,  reg=low     : v2_regprobe_fold0_lowdrop      train_max=0.758
#   h=128, reg=default : v2_higher_capacity             train_max=0.751
#   h=128, reg=low     : THIS RUN, v2_caprobe_h128_fold0
#   h=256, reg=low     : THIS RUN, v2_caprobe_h256_fold0
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

submit_capacity() {
    local hidden="$1"
    local save_dir="runs/multimodal_features/v2_caprobe_h${hidden}_fold0"
    echo ""
    echo "--- Submitting h=$hidden → $save_dir ---"
    # Subshell so per-job env vars don't leak
    (
        export MM_ARCH="v2"
        export MM_VIDEO_FEATURE_DIR="data/dinov2_features_meanpatch"
        export MM_AUDIO_FEATURE_DIR="data/wavlm_baseplus_features"
        export MM_SAVE_DIR="$save_dir"
        export MM_VIDEO_HIDDEN="$hidden"
        export MM_AUDIO_HIDDEN="$hidden"
        export MM_HEAD_HIDDEN="$hidden"
        export MM_DROPOUT="0.05"
        export MM_WEIGHT_DECAY="0.0"
        export MM_EPOCHS="30"
        export MM_PATIENCE="30"
        export MM_NUM_FOLDS="5"
        export MM_FOLD_IDX="0"
        export MM_EARLY_STOP_METRIC="val_loss"
        sh "$LAUNCHER"
    )
}

echo "=== [$SCRIPT_VERSION] Submitting capacity probe (fold 0, low-reg, h=128 + h=256) ==="

submit_capacity 128
submit_capacity 256

echo ""
echo "=== Both jobs submitted. Check status: bjobs ==="
echo "Once both complete, compare the capacity curve by running:"
echo "  bash scripts/bsub/capacity_probe_summary.sh"

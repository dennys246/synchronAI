#!/bin/bash
SCRIPT_VERSION="submit_audioshuffle_probe_fold0-v1"
# =============================================================================
# Probe 2: within-recording audio-shuffle augmentation.
#
# Hypothesis: the audio pathway in v2 collapses into a video-redundant
# representation (R²(A→V) ≥ 0.92 across all conditions tested). With a fused
# BCE loss, gradient descent prefers the easy local minimum where audio
# just mirrors video — adding nothing new and being unbreakable by capacity
# or regularization changes.
#
# This probe attacks the redundancy directly via data augmentation: during
# training, each sample has p=0.3 probability of having its audio swapped
# with audio from a different second IN THE SAME RECORDING, with the label
# forced to 0. This forces the model to attend to actual cross-modal
# temporal alignment (because that's the only thing that distinguishes
# real synchronous pairs from shuffled-within-recording pairs) instead of
# just encoding scene/subject identity in both pathways.
#
# Constraint: WITHIN-RECORDING shuffling only. Cross-recording shuffles
# would teach the model to detect different scene/subject — a trivial
# signal unrelated to synchrony. By keeping the same recording, the only
# thing that changes is the audio's temporal position, isolating the
# synchrony signal.
#
# Decision rule (after 30 epochs):
#   - val_AUC breaks past 0.80 → augmentation broke the redundancy and
#     unlocked real audio contribution. Run full 5-fold CV.
#   - val_AUC stays at 0.72 + train_acc plateaus → augmentation didn't
#     help; the audio pathway either still collapses or the audio just
#     doesn't carry independent synchrony signal at this granularity.
#   - val_AUC drops (overfits to "spot the shuffle") → p=0.3 is too high
#     or the within-recording constraint isn't tight enough; consider
#     same-dyad-different-timepoint shuffles instead.
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

export MM_ARCH="v2"
export MM_VIDEO_FEATURE_DIR="data/dinov2_features_meanpatch"
export MM_AUDIO_FEATURE_DIR="data/wavlm_baseplus_features"
export MM_SAVE_DIR="runs/multimodal_features/v2_audioshuffle03_fold0"
export MM_DROPOUT="0.05"
export MM_WEIGHT_DECAY="0.0"
export MM_EPOCHS="30"
export MM_PATIENCE="30"
export MM_NUM_FOLDS="5"
export MM_FOLD_IDX="0"
export MM_EARLY_STOP_METRIC="val_loss"
# The actual probe knob.
export MM_AUDIO_SHUFFLE_PROB="0.3"

echo "=== [$SCRIPT_VERSION] Submitting audio-shuffle probe (fold 0, p=0.3) ==="
echo "  save_dir:           $MM_SAVE_DIR"
echo "  audio_shuffle_prob: $MM_AUDIO_SHUFFLE_PROB (within-recording only)"
echo "  dropout / wd:       $MM_DROPOUT / $MM_WEIGHT_DECAY (matches probe 1 reg)"
echo "  epochs / patience:  $MM_EPOCHS / $MM_PATIENCE (no early stop)"
echo "  fold:               $MM_FOLD_IDX / $MM_NUM_FOLDS"

sh "$LAUNCHER"

echo ""
echo "=== Submitted. Check status: bjobs ==="
echo "Compare against the regprobe baseline (same fold, same reg, no shuffle):"
echo "  python3 -c \"import json;"
echo "    a=json.load(open('runs/multimodal_features/v2_audioshuffle03_fold0/history.json'));"
echo "    b=json.load(open('runs/multimodal_features/v2_regprobe_fold0_lowdrop/history.json'));"
echo "    print('shuffle  train:', [round(x,3) for x in a['train_accs']]);"
echo "    print('shuffle  val:  ', [round(x,3) for x in a['val_accs']]);"
echo "    print('shuffle  auc:  ', [round(x,3) for x in a['val_aucs']]);"
echo "    print('baseline train:', [round(x,3) for x in b['train_accs']]);"
echo "    print('baseline val:  ', [round(x,3) for x in b['val_accs']]);"
echo "    print('baseline auc:  ', [round(x,3) for x in b['val_aucs']])\""

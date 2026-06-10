#!/bin/bash
SCRIPT_VERSION="submit_regprobe_fold0-v1"
# =============================================================================
# Regularization probe: is the ~0.72 val_AUC ceiling data-limited or
# feature/label-limited?
#
# v2_baseline_v6 hits train_acc=0.746 at epoch 15 (still climbing) while val
# peaks at ep4 (0.798) and decays to 0.703 by ep15 — textbook overfitting.
# But the run uses dropout=0.3 in three places + WD=1e-2, so it's unclear
# whether train_acc is being *held down* by regularization or genuinely
# can't fit past 0.75. This probe drops dropout to 0.05 and WD to 0 on
# fold 0 and disables early stopping so the train trajectory plays out.
#
# Decision rule (after 30 epochs):
#   - train_acc > 0.85 + val plateau ≤ 0.76 → data-limited (scale subjects)
#   - train_acc ≤ 0.75 even unregularized   → feature/label-limited
#   - in between                             → partial; data helps modestly
#
# Control: runs/multimodal_features/v2_baseline_v6_cv5/fold_0 (already on disk).
# Same fold, same seed, default reg, ran 15 epochs.
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

export MM_ARCH="v2"
export MM_VIDEO_FEATURE_DIR="data/dinov2_features_meanpatch"
export MM_AUDIO_FEATURE_DIR="data/wavlm_baseplus_features"
export MM_SAVE_DIR="runs/multimodal_features/v2_regprobe_fold0_lowdrop"
export MM_DROPOUT="0.05"
export MM_WEIGHT_DECAY="0.0"
export MM_EPOCHS="30"
export MM_PATIENCE="30"
export MM_NUM_FOLDS="5"
export MM_FOLD_IDX="0"
export MM_EARLY_STOP_METRIC="val_loss"

echo "=== [$SCRIPT_VERSION] Submitting regularization probe (fold 0, low-reg) ==="
echo "  save_dir:   $MM_SAVE_DIR"
echo "  dropout:    $MM_DROPOUT  (control: 0.3)"
echo "  wd:         $MM_WEIGHT_DECAY  (control: 1e-2)"
echo "  epochs:     $MM_EPOCHS  patience: $MM_PATIENCE  (no early stop)"
echo "  fold:       $MM_FOLD_IDX / $MM_NUM_FOLDS"

sh "$LAUNCHER"

echo ""
echo "=== Submitted. Check status: bjobs ==="
echo "Once complete, compare train/val curves against the existing control:"
echo "  python -c \"import json; from pathlib import Path; \\"
echo "    a=json.load(open('$MM_SAVE_DIR/history.json')); \\"
echo "    b=json.load(open('runs/multimodal_features/v2_baseline_v6_cv5/fold_0/history.json')); \\"
echo "    print('low-reg train_accs:', [round(x,3) for x in a['train_accs']]); \\"
echo "    print('low-reg val_accs  :', [round(x,3) for x in a['val_accs']]); \\"
echo "    print('control train_accs:', [round(x,3) for x in b['train_accs']]); \\"
echo "    print('control val_accs  :', [round(x,3) for x in b['val_accs']])\""

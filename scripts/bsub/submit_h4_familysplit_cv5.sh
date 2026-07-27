#!/bin/bash
SCRIPT_VERSION="submit_h4_familysplit_cv5-v1"
# =============================================================================
# H4 re-test under the leakage-corrected (family_id) split.
#
# Submits two 5-fold CV arms that differ in exactly one variable — the audio
# feature dir — so the H4 question ("is WavLM the wrong audio representation?")
# gets a clean answer:
#
#   arm A: data/wavlm_baseplus_features   (49, 768)  — the incumbent
#   arm B: data/prosodic_features         (96,  25)  — openSMILE eGeMAPS
#
# Why re-run something already on disk:
#   runs/multimodal_features/v2_prosodic_cv5 already answers H4 (negative:
#   prosodic tracked the WavLM/video-only null to within ~0.003 AUC), but that
#   run and its baselines all predate two fixes. Their config.json has neither
#   a `group_key` nor a `fallback_csv` field, i.e. they were split on
#   subject_id — which leaks CARE dyads, since a parent (4-digit) and child
#   (5-digit) share a family_id and can land on opposite sides of the split.
#
#   The comparison between them is still internally valid (all four arms share
#   the same confound), but the absolute numbers are inflated, and H4 should
#   not stay closed on a leaky split when re-testing costs ~9 small CPU jobs.
#
#   runs/multimodal_features/family_cv is the only family-split 5-fold on disk
#   and it is INCOMPLETE — fold0 only, folds 1-4 never ran. So arm A is not
#   redundant either; it completes that baseline.
#
# Config mirrors family_cv/fold0 exactly (v2, 64/64/64, dropout 0.3, wd 1e-2,
# lr 5e-5, 30 epochs, patience 10, val_loss early stop) so arm A is a
# continuation of that run rather than a new condition. Seed is not settable
# from the launcher — multimodal_from_features_bsub.sh never passes --seed, so
# the trainer default of 42 applies, which is what family_cv used. fallback_csv stays
# unset to match; the bogus-segment filter is a separate variable and folding
# it in here would confound the arms.
#
# Writes to fresh save dirs. Does NOT write into family_cv/ — re-running a
# launcher over a completed run overwrites its checkpoints, and fold0 there is
# the only family-split result we have.
#
# Usage:
#   bash scripts/bsub/submit_h4_familysplit_cv5.sh
#
# Skips any fold whose best.pt already exists, so it is safe to re-run after a
# partial failure.
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"
NUM_FOLDS=5
SAVE_DIR_BASE="runs/multimodal_features/h4_familysplit_cv5"

WAVLM_DIR="data/wavlm_baseplus_features"
PROSODIC_DIR="data/prosodic_features"
VIDEO_DIR="data/dinov2_features_meanpatch"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

for d in "$WAVLM_DIR" "$PROSODIC_DIR" "$VIDEO_DIR"; do
    if [ ! -f "$d/feature_index.csv" ]; then
        echo "ERROR: feature index missing — $d/feature_index.csv"
        exit 1
    fi
done

echo "=== [$SCRIPT_VERSION] H4 re-test, family_id split, ${NUM_FOLDS}-fold ==="
echo "  save_dir_base: $SAVE_DIR_BASE"
echo "  video:         $VIDEO_DIR"
echo "  arm A audio:   $WAVLM_DIR"
echo "  arm B audio:   $PROSODIC_DIR"
echo "  group_key:     family_id"

n_submitted=0
n_skipped=0

submit_fold() {
    local arm="$1"
    local audio_dir="$2"
    local fold="$3"
    local save_dir="${SAVE_DIR_BASE}/${arm}/fold_${fold}"

    if [ -f "$save_dir/best.pt" ]; then
        echo "--- skip ${arm} fold ${fold}: $save_dir/best.pt exists ---"
        n_skipped=$((n_skipped + 1))
        return
    fi

    echo ""
    echo "--- Submitting ${arm} fold ${fold}/${NUM_FOLDS} → $save_dir ---"
    (
        export MM_ARCH="v2"
        export MM_VIDEO_FEATURE_DIR="$VIDEO_DIR"
        export MM_AUDIO_FEATURE_DIR="$audio_dir"
        export MM_SAVE_DIR="$save_dir"
        export MM_VIDEO_HIDDEN="64"
        export MM_AUDIO_HIDDEN="64"
        export MM_HEAD_HIDDEN="64"
        export MM_DROPOUT="0.3"
        export MM_WEIGHT_DECAY="1e-2"
        export MM_LEARNING_RATE="5e-5"
        export MM_BATCH_SIZE="128"
        export MM_EPOCHS="30"
        export MM_PATIENCE="10"
        export MM_WARMUP_EPOCHS="5"
        export MM_NUM_FOLDS="$NUM_FOLDS"
        export MM_FOLD_IDX="$fold"
        export MM_EARLY_STOP_METRIC="val_loss"
        export MM_GROUP_KEY="family_id"
        sh "$LAUNCHER"
    )
    n_submitted=$((n_submitted + 1))
}

for fold in 0 1 2 3 4; do
    submit_fold "wavlm" "$WAVLM_DIR" "$fold"
done

for fold in 0 1 2 3 4; do
    submit_fold "prosodic" "$PROSODIC_DIR" "$fold"
done

echo ""
echo "=== Submitted $n_submitted job(s), skipped $n_skipped already-complete fold(s). ==="
echo "Status: bjobs"
echo ""
echo "Once all folds finish, aggregate each arm separately:"
echo "  python scripts/aggregate_kfold_results.py --cv-dir $SAVE_DIR_BASE/wavlm"
echo "  python scripts/aggregate_kfold_results.py --cv-dir $SAVE_DIR_BASE/prosodic"

#!/bin/bash
SCRIPT_VERSION="wavlm_audio_sweep_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 16000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>16GB] rusage[mem=16GB]'

# =============================================================================
# WavLM Audio Feature Training — Single Sweep Run (training only)
#
# Lightweight training-only job on pre-extracted WavLM .pt features.
# Does NOT need GPU or transformers — just torch + sklearn.
#
# Features must already be extracted by wavlm_extract_bsub.sh.
# The pre_wavlm_audio_sweep_bsub.sh submission script handles dependency.
#
# Config is passed via environment variables:
#   SWEEP_RUN_NAME, SWEEP_ENCODER, SWEEP_FEATURE_DIR,
#   SWEEP_TEMPORAL_AGG, SWEEP_HIDDEN_DIM, SWEEP_DROPOUT,
#   SWEEP_LEARNING_RATE, SWEEP_WEIGHT_DECAY, SWEEP_PATIENCE,
#   SWEEP_LABEL_SMOOTHING, SWEEP_MIXUP_ALPHA
# =============================================================================

source $SYNCHRONAI_DIR/ml-env/bin/activate

cd $SYNCHRONAI_DIR

# =============================================================================
# Sweep Run Configuration (from environment variables)
# =============================================================================

RUN_NAME="${SWEEP_RUN_NAME:-baseline}"
ENCODER="${SWEEP_ENCODER:-wavlm-base-plus}"
FEATURE_DIR="${SWEEP_FEATURE_DIR:-data/wavlm_baseplus_features}"
TEMPORAL_AGG="${SWEEP_TEMPORAL_AGG:-mean}"
HIDDEN_DIM="${SWEEP_HIDDEN_DIM:-128}"
DROPOUT="${SWEEP_DROPOUT:-0.5}"
LEARNING_RATE="${SWEEP_LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${SWEEP_WEIGHT_DECAY:-1e-3}"
PATIENCE="${SWEEP_PATIENCE:-15}"
LABEL_SMOOTHING="${SWEEP_LABEL_SMOOTHING:-0.0}"
MIXUP_ALPHA="${SWEEP_MIXUP_ALPHA:-0.0}"
PROJECT_DIM="${SWEEP_PROJECT_DIM:-0}"
OUTPUT_DIR="${SWEEP_OUTPUT_BASE:-runs/audio_sweep}/${RUN_NAME}"

echo "=== [$SCRIPT_VERSION] ==="
echo ""
echo "=========================================="
echo "  Audio Sweep Run: ${RUN_NAME}"
echo "=========================================="
echo "  Encoder:          ${ENCODER}"
echo "  Feature dir:      ${FEATURE_DIR}"
echo "  Temporal agg:     ${TEMPORAL_AGG}"
echo "  Hidden dim:       ${HIDDEN_DIM}"
echo "  Dropout:          ${DROPOUT}"
echo "  Learning rate:    ${LEARNING_RATE}"
echo "  Weight decay:     ${WEIGHT_DECAY}"
echo "  Patience:         ${PATIENCE}"
echo "  Label smoothing:  ${LABEL_SMOOTHING}"
echo "  Mixup alpha:      ${MIXUP_ALPHA}"
echo "  Project dim:      ${PROJECT_DIM}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "=========================================="

# Verify features exist
if [ ! -f "${FEATURE_DIR}/feature_index.csv" ]; then
    echo "ERROR: Feature index not found at ${FEATURE_DIR}/feature_index.csv"
    echo "Run the extraction job first (wavlm_extract_bsub.sh)"
    exit 1
fi

# =============================================================================
# Train on pre-extracted features
# =============================================================================

echo ""
echo "=== Training ${RUN_NAME} ==="

python scripts/train_audio_from_features.py \
    --feature-dir "${FEATURE_DIR}" \
    --save-dir "${OUTPUT_DIR}" \
    --encoder "${ENCODER}" \
    --temporal-aggregation "${TEMPORAL_AGG}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --batch-size 64 \
    --epochs 50 \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --label-smoothing "${LABEL_SMOOTHING}" \
    --mixup-alpha "${MIXUP_ALPHA}" \
    --project-dim "${PROJECT_DIM}" \
    --seed 42 \
    --num-workers 0

echo ""
echo "=== Audio sweep run ${RUN_NAME} complete ==="
echo "Results: ${OUTPUT_DIR}/history.json"

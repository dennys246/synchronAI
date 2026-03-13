#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 16000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>16GB] rusage[mem=16GB]'

# =============================================================================
# DINOv2 Feature Training — Single Sweep Run (training only)
#
# This is a lightweight training-only job that operates on pre-extracted .pt
# feature files. It does NOT need opencv or GPU — just torch + sklearn.
#
# Features must already be extracted by the dinov2_extract_bsub.sh job.
# The pre_dinov2_sweep_bsub.sh submission script handles the dependency.
#
# Config is passed via environment variables:
#   SWEEP_RUN_NAME, SWEEP_BACKBONE, SWEEP_FEATURE_DIR,
#   SWEEP_TEMPORAL_AGG, SWEEP_HIDDEN_DIM, SWEEP_DROPOUT,
#   SWEEP_LEARNING_RATE, SWEEP_WEIGHT_DECAY, SWEEP_PATIENCE
# =============================================================================

source $SYNCHRONAI_DIR/ml-env/bin/activate

cd $SYNCHRONAI_DIR

# No dependency installation needed — ml-env was built by dinov2_extract_bsub.sh

# =============================================================================
# Sweep Run Configuration (from environment variables)
# =============================================================================

RUN_NAME="${SWEEP_RUN_NAME:-baseline}"
BACKBONE="${SWEEP_BACKBONE:-dinov2-base}"
FEATURE_DIR="${SWEEP_FEATURE_DIR:-data/dinov2_features_meanpatch}"
TEMPORAL_AGG="${SWEEP_TEMPORAL_AGG:-lstm}"
HIDDEN_DIM="${SWEEP_HIDDEN_DIM:-128}"
DROPOUT="${SWEEP_DROPOUT:-0.5}"
LEARNING_RATE="${SWEEP_LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${SWEEP_WEIGHT_DECAY:-1e-3}"
PATIENCE="${SWEEP_PATIENCE:-10}"
OUTPUT_DIR="runs/dinov2_sweep/${RUN_NAME}"

echo ""
echo "=========================================="
echo "  Sweep Run: ${RUN_NAME}"
echo "=========================================="
echo "  Backbone:       ${BACKBONE}"
echo "  Feature dir:    ${FEATURE_DIR}"
echo "  Temporal agg:   ${TEMPORAL_AGG}"
echo "  Hidden dim:     ${HIDDEN_DIM}"
echo "  Dropout:        ${DROPOUT}"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Weight decay:   ${WEIGHT_DECAY}"
echo "  Patience:       ${PATIENCE}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "=========================================="

# Verify features exist
if [ ! -f "${FEATURE_DIR}/feature_index.csv" ]; then
    echo "ERROR: Feature index not found at ${FEATURE_DIR}/feature_index.csv"
    echo "Run the extraction job first (dinov2_extract_bsub.sh)"
    exit 1
fi

# =============================================================================
# Train on pre-extracted features
# =============================================================================

echo ""
echo "=== Training ${RUN_NAME} ==="

python scripts/train_from_features.py \
    --feature-dir "${FEATURE_DIR}" \
    --save-dir "${OUTPUT_DIR}" \
    --backbone "${BACKBONE}" \
    --temporal-aggregation "${TEMPORAL_AGG}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --batch-size 64 \
    --epochs 50 \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --seed 42 \
    --num-workers 0

echo ""
echo "=== Sweep run ${RUN_NAME} complete ==="
echo "Results: ${OUTPUT_DIR}/history.json"
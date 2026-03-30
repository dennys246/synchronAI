#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 16000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>16GB] rusage[mem=16GB]'

# =============================================================================
# DINOv2 Progressive Feature Training — Single Sweep Run
#
# Lightweight training-only job on pre-extracted features at multiple
# resolutions. Does NOT need GPU — just torch + sklearn.
#
# Multi-resolution features must already exist (dinov2_extract_multirez_bsub.sh).
#
# Config is passed via environment variables:
#   SWEEP_RUN_NAME, SWEEP_MODE (progressive|flat_mixup|flat_restarts|flat_baseline),
#   SWEEP_TEMPORAL_AGG, SWEEP_HIDDEN_DIM, SWEEP_DROPOUT,
#   SWEEP_LEARNING_RATE, SWEEP_WEIGHT_DECAY, SWEEP_PATIENCE,
#   SWEEP_MIXUP_ALPHA, SWEEP_LR_SCHEDULE, SWEEP_LR_RESTART_PERIOD
# =============================================================================

source $SYNCHRONAI_DIR/ml-env/bin/activate

cd $SYNCHRONAI_DIR

# Ensure both project root and scripts/ are importable
export PYTHONPATH="$SYNCHRONAI_DIR:$SYNCHRONAI_DIR/scripts:$PYTHONPATH"

# =============================================================================
# Sweep Configuration (from environment)
# =============================================================================

RUN_NAME="${SWEEP_RUN_NAME:-prog_baseline}"
MODE="${SWEEP_MODE:-progressive}"
TEMPORAL_AGG="${SWEEP_TEMPORAL_AGG:-lstm}"
HIDDEN_DIM="${SWEEP_HIDDEN_DIM:-128}"
DROPOUT="${SWEEP_DROPOUT:-0.7}"
LEARNING_RATE="${SWEEP_LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${SWEEP_WEIGHT_DECAY:-1e-2}"
PATIENCE="${SWEEP_PATIENCE:-15}"
MIXUP_ALPHA="${SWEEP_MIXUP_ALPHA:-0.2}"
LR_SCHEDULE="${SWEEP_LR_SCHEDULE:-cosine_restarts}"
LR_RESTART_PERIOD="${SWEEP_LR_RESTART_PERIOD:-10}"
OUTPUT_DIR="runs/dinov2_progressive/${RUN_NAME}"

# Feature directories
FEAT_112="data/dinov2_features_small_112"
FEAT_168="data/dinov2_features_small_168"
FEAT_224="data/dinov2_features_small_meanpatch"

echo ""
echo "=========================================="
echo "  Progressive Sweep Run: ${RUN_NAME}"
echo "=========================================="
echo "  Mode:           ${MODE}"
echo "  Temporal agg:   ${TEMPORAL_AGG}"
echo "  Hidden dim:     ${HIDDEN_DIM}"
echo "  Dropout:        ${DROPOUT}"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Weight decay:   ${WEIGHT_DECAY}"
echo "  Patience:       ${PATIENCE}"
echo "  Mixup alpha:    ${MIXUP_ALPHA}"
echo "  LR schedule:    ${LR_SCHEDULE}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "=========================================="

if [ "$MODE" = "progressive" ]; then
    # Progressive resolution training (112 → 168 → 224)
    python scripts/train_progressive_features.py \
        --feature-dirs "${FEAT_112}" "${FEAT_168}" "${FEAT_224}" \
        --stage-epochs 10 10 15 \
        --stage-lrs 3e-4 1e-4 "${LEARNING_RATE}" \
        --save-dir "${OUTPUT_DIR}" \
        --backbone dinov2-small \
        --temporal-aggregation "${TEMPORAL_AGG}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --dropout "${DROPOUT}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --patience "${PATIENCE}" \
        --mixup-alpha "${MIXUP_ALPHA}" \
        --lr-schedule "${LR_SCHEDULE}" \
        --lr-restart-period "${LR_RESTART_PERIOD}" \
        --batch-size 64 \
        --seed 42 \
        --num-workers 0
else
    # Flat training (single resolution, for ablation comparison)
    python scripts/train_from_features.py \
        --feature-dir "${FEAT_224}" \
        --save-dir "${OUTPUT_DIR}" \
        --backbone dinov2-small \
        --temporal-aggregation "${TEMPORAL_AGG}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --dropout "${DROPOUT}" \
        --batch-size 64 \
        --epochs 35 \
        --learning-rate "${LEARNING_RATE}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --patience "${PATIENCE}" \
        --mixup-alpha "${MIXUP_ALPHA}" \
        --lr-schedule "${LR_SCHEDULE}" \
        --lr-restart-period "${LR_RESTART_PERIOD}" \
        --seed 42 \
        --num-workers 0
fi

echo ""
echo "=== Sweep run ${RUN_NAME} complete ==="
echo "Results: ${OUTPUT_DIR}/history.json"
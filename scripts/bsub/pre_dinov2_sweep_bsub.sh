#!/bin/sh
# =============================================================================
# Submit DINOv2 Small Backbone Sweep — Focused on dinov2-small (384-dim)
#
# The initial sweep found signal with dinov2-small (val AUC 0.692, val acc 0.669)
# but with significant overfitting (train acc 0.81 vs val 0.67). This sweep
# explores regularization, capacity, and aggregation variants around that config.
#
# Features and ml-env already exist — submits training jobs directly.
#
# After all jobs complete, run:
#   python scripts/compare_sweep_results.py --sweep-dir runs/dinov2_sweep
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
SWEEP_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/dinov2_sweep_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

SMALL_FEATURES="data/dinov2_features_small_meanpatch"

echo "=========================================="
echo "  DINOv2 Small Backbone Sweep"
echo "  All runs use dinov2-small (384-dim)"
echo "  Date: $DATE"
echo "=========================================="
echo ""
echo "Submitting 5 training jobs (features + ml-env already exist)..."

# --- Run 1: small_baseline — reproduce the original promising run ---
export SWEEP_RUN_NAME="small_baseline"
export SWEEP_BACKBONE="dinov2-small"
export SWEEP_FEATURE_DIR="$SMALL_FEATURES"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_PATIENCE="10"

echo "  1/5: ${SWEEP_RUN_NAME} (LSTM h=128, d=0.5, lr=3e-5, wd=1e-3) — original config"
bsub -J "synchronai-sweep-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/sweep \
     < "$SWEEP_SCRIPT"

# --- Run 2: small_heavy_reg — max regularization to fight overfitting ---
export SWEEP_RUN_NAME="small_heavy_reg"
export SWEEP_BACKBONE="dinov2-small"
export SWEEP_FEATURE_DIR="$SMALL_FEATURES"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="10"

echo "  2/5: ${SWEEP_RUN_NAME} (LSTM h=128, d=0.7, wd=1e-2) — heavy regularization"
bsub -J "synchronai-sweep-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/sweep \
     < "$SWEEP_SCRIPT"

# --- Run 3: small_lowcap — smaller model + slower learning ---
export SWEEP_RUN_NAME="small_lowcap"
export SWEEP_BACKBONE="dinov2-small"
export SWEEP_FEATURE_DIR="$SMALL_FEATURES"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.6"
export SWEEP_LEARNING_RATE="1e-5"
export SWEEP_WEIGHT_DECAY="5e-3"
export SWEEP_PATIENCE="12"

echo "  3/5: ${SWEEP_RUN_NAME} (LSTM h=64, d=0.6, lr=1e-5, wd=5e-3) — low capacity + slow"
bsub -J "synchronai-sweep-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/sweep \
     < "$SWEEP_SCRIPT"

# --- Run 4: small_attention — attention pooling instead of LSTM ---
export SWEEP_RUN_NAME="small_attention"
export SWEEP_BACKBONE="dinov2-small"
export SWEEP_FEATURE_DIR="$SMALL_FEATURES"
export SWEEP_TEMPORAL_AGG="attention"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.6"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="5e-3"
export SWEEP_PATIENCE="10"

echo "  4/5: ${SWEEP_RUN_NAME} (attention h=128, d=0.6, wd=5e-3) — less temporal params"
bsub -J "synchronai-sweep-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/sweep \
     < "$SWEEP_SCRIPT"

# --- Run 5: small_conservative — minimal capacity, maximum constraint ---
export SWEEP_RUN_NAME="small_conservative"
export SWEEP_BACKBONE="dinov2-small"
export SWEEP_FEATURE_DIR="$SMALL_FEATURES"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="1e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="12"

echo "  5/5: ${SWEEP_RUN_NAME} (mean pool h=64, d=0.7, lr=1e-5, wd=1e-2) — minimal"
bsub -J "synchronai-sweep-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/sweep \
     < "$SWEEP_SCRIPT"

echo ""
echo "=========================================="
echo "  5 training jobs submitted"
echo "  All training on dinov2-small (384-dim)"
echo ""
echo "  Monitor with: bjobs -g /$USER/sweep"
echo ""
echo "  After all complete:"
echo "    python scripts/compare_sweep_results.py"
echo "=========================================="

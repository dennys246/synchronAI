#!/bin/bash
SCRIPT_VERSION="pre_fnirs_child_adult_sweep_tile-v1"
# =============================================================================
# fNIRS Child/Adult Classification Sweep — TILE re-run (Option B), Training Only
#
# Reads tiled-windowing features from:
#   $SYNCHRONAI_DIR/data/fnirs_perpair_{size}_tile_features/
# Writes to fresh:
#   $SYNCHRONAI_DIR/runs/fnirs_child_adult_sweep_tile/{size}_{classifier}/
# so the existing sweep results stay intact until we swap. Mirrors
# pre_fnirs_child_adult_sweep_bsub.sh with _tile paths; 5 classifiers per size.
#
# Run AFTER: pre_fnirs_extract_features_tile_bsub.sh
# =============================================================================

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

SWEEP_DIR="$SYNCHRONAI_DIR/runs/fnirs_child_adult_sweep_tile"

echo "=========================================="
echo "  [$SCRIPT_VERSION]  fNIRS Child/Adult Sweep (TILE)"
echo "  Date: $DATE"
echo "=========================================="

submit_model_sweep() {
    local MODEL_NAME="$1"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_tile_features"

    # Memory scales with the mmap'd packed feature file (random-shuffled batches
    # fault pages in toward the file size). Same per-size caps as the original.
    local MEM_GB
    case "$MODEL_NAME" in
        micro)  MEM_GB=16 ;;
        small)  MEM_GB=32 ;;
        medium) MEM_GB=48 ;;
        large)  MEM_GB=96 ;;
        *)      MEM_GB=32 ;;
    esac
    local MEM_KB=$((MEM_GB * 1024 * 1024))

    echo ""
    echo "=== $MODEL_NAME (tile, mem=${MEM_GB}GB) ==="

    bsub -J "synchronai-sweep-tile-${MODEL_NAME}-$DATE" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M $MEM_KB \
         -a 'docker(continuumio/anaconda3)' \
         -n 4 \
         -R "select[mem>${MEM_GB}GB] rusage[mem=${MEM_GB}GB] span[hosts=1]" \
         -oo "$LOG_DIR/fnirs_sweep_tile_${MODEL_NAME}_$DATE.log" \
         -g /$USER/fnirs_sweep_tile \
         << SWEEP_EOF
echo "=== [$SCRIPT_VERSION] $MODEL_NAME ==="
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONNOUSERSITE=1
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"
echo "ML_PY=\$ML_PY"
"\$ML_PY" -c "import sys, torch, synchronai; print('python:', sys.executable, 'torch:', torch.__version__)" || {
    echo "FATAL: ml-env import check failed — aborting before training"
    exit 2
}

if [ ! -f "$FEATURE_DIR/feature_index.csv" ]; then
    echo "ERROR: Features not found at $FEATURE_DIR/feature_index.csv — run tile extraction first."
    exit 1
fi

train_classifier() {
    local RUN_NAME="\$1"
    local HIDDEN_DIM="\$2"
    local DROPOUT="\$3"
    local POOL="\$4"
    local LR="\$5"
    local SAVE_DIR="$SWEEP_DIR/${MODEL_NAME}_\${RUN_NAME}"

    echo ""
    echo "=== Training ${MODEL_NAME}_\${RUN_NAME} (h=\${HIDDEN_DIM}, pool=\${POOL}) ==="
    if [ -f "\$SAVE_DIR/best.pt" ]; then
        echo "SKIP: \$SAVE_DIR/best.pt already exists — delete to force retrain"
        return 0
    fi

    "\$ML_PY" scripts/train_fnirs_from_features.py \
        --feature-dir "$FEATURE_DIR" \
        --save-dir "\$SAVE_DIR" \
        --label-column participant_type \
        --label-map "child:0,adult:1" \
        --hidden-dim \$HIDDEN_DIM \
        --dropout \$DROPOUT \
        --pool \$POOL \
        --learning-rate \$LR \
        --weight-decay 1e-2 \
        --warmup-epochs 3 \
        --patience 15 \
        --epochs 50 \
        --batch-size 32 \
        --num-workers 0 \
        --seed 42 \
        --include-tiers "gold,standard" \
        --holdout-tiers "gold,salvageable"
    local PY_RC=\$?
    if [ \$PY_RC -ne 0 ]; then
        echo "FAILED: ${MODEL_NAME}_\${RUN_NAME} exit code \$PY_RC"
    fi
}

train_classifier "linear"     0   0.0 "mean" "1e-3"
train_classifier "mlp32"      32  0.3 "mean" "3e-4"
train_classifier "mlp64_proj" 64  0.5 "mean" "3e-4"
train_classifier "lstm64"     64  0.3 "lstm" "3e-4"
train_classifier "lstm_proj"  64  0.5 "lstm" "3e-4"

echo ""
echo "=== All classifiers complete for $MODEL_NAME (tile) ==="
SWEEP_EOF
}

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("micro" "small" "medium" "large")
fi

for MODEL in "${MODELS[@]}"; do
    submit_model_sweep "$MODEL"
done

echo ""
echo "=========================================="
echo "  ${#MODELS[@]} tile training job(s) submitted"
echo "  Monitor: bjobs -g /\$USER/fnirs_sweep_tile"
echo "  Outputs: runs/fnirs_child_adult_sweep_tile/{size}_{classifier}/"
echo "=========================================="

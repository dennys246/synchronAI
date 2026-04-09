#!/bin/bash
SCRIPT_VERSION="pre_fnirs_child_adult_sweep-v7"
# =============================================================================
# fNIRS Child/Adult Classification Sweep — Training Only
#
# Assumes features are already extracted to:
#   $SYNCHRONAI_DIR/data/fnirs_perpair_{micro,small,medium,large}_features/
#
# For each model size, trains 5 classifiers sequentially in one job.
# Total: 4 jobs (1 per model size)
# =============================================================================

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

SWEEP_DIR="$SYNCHRONAI_DIR/runs/fnirs_child_adult_sweep"

echo "=========================================="
echo "  [$SCRIPT_VERSION]"
echo "  fNIRS Child/Adult Sweep — Training Only"
echo "  Date: $DATE"
echo "=========================================="

submit_model_sweep() {
    local MODEL_NAME="$1"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_features"

    echo ""
    echo "=== $MODEL_NAME ==="

    bsub -J "synchronai-sweep-${MODEL_NAME}-$DATE" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 8000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 4 \
         -R 'select[mem>8GB] rusage[mem=8GB]' \
         -oo "$LOG_DIR/fnirs_sweep_${MODEL_NAME}_$DATE.log" \
         -g /$USER/fnirs_sweep \
         << SWEEP_EOF
echo "=== [$SCRIPT_VERSION] $MODEL_NAME ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"

if [ ! -f "$FEATURE_DIR/feature_index.csv" ]; then
    echo "ERROR: Features not found at $FEATURE_DIR/feature_index.csv"
    echo "Run feature extraction first."
    exit 1
fi

train_classifier() {
    local RUN_NAME="\$1"
    local HIDDEN_DIM="\$2"
    local DROPOUT="\$3"
    local POOL="\$4"
    local LR="\$5"

    echo ""
    echo "=== Training ${MODEL_NAME}_\${RUN_NAME} (h=\${HIDDEN_DIM}, pool=\${POOL}) ==="

    python scripts/train_fnirs_from_features.py \
        --feature-dir "$FEATURE_DIR" \
        --save-dir "$SWEEP_DIR/${MODEL_NAME}_\${RUN_NAME}" \
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
}

train_classifier "linear"     0   0.0 "mean" "1e-3"
train_classifier "mlp32"      32  0.3 "mean" "3e-4"
train_classifier "mlp64_proj" 64  0.5 "mean" "3e-4"
train_classifier "lstm64"     64  0.3 "lstm" "3e-4"
train_classifier "lstm_proj"  64  0.5 "lstm" "3e-4"

echo ""
echo "=== All classifiers complete for $MODEL_NAME ==="
SWEEP_EOF
}

submit_model_sweep "micro"
submit_model_sweep "small"
submit_model_sweep "medium"
submit_model_sweep "large"

echo ""
echo "=========================================="
echo "  4 training jobs submitted"
echo "  Monitor: bjobs -g /\$USER/fnirs_sweep"
echo "=========================================="

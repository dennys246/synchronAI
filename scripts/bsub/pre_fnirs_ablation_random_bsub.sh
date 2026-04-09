#!/bin/bash
SCRIPT_VERSION="pre_fnirs_ablation_random_bsub-v6"
# =============================================================================
# fNIRS Ablation: Random Per-Pair Encoder — Training Only
#
# Assumes random features are already extracted to:
#   $SYNCHRONAI_DIR/data/fnirs_perpair_{MODEL}_random_features/
#
# Single job: trains lstm64 on random features.
# =============================================================================

MODEL_NAME="${PERPAIR_MODEL:-large}"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true
export ABLATION_MODEL_NAME="$MODEL_NAME"

DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

echo "=== [$SCRIPT_VERSION] ==="
echo "  Ablation: $MODEL_NAME (random encoder)"

bsub -J "synchronai-fnirs-random-${MODEL_NAME}-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 8000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 4 \
     -R 'select[mem>8GB] rusage[mem=8GB]' \
     -oo "$LOG_DIR/fnirs_ablation_random_${MODEL_NAME}_$DATE.log" \
     -g /$USER/fnirs_ablation \
     << 'ABLATION_EOF'
echo "=== [pre_fnirs_ablation_random_bsub-v6] ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

RANDOM_FEATURES="$SYNCHRONAI_DIR/data/fnirs_perpair_${ABLATION_MODEL_NAME}_random_features"
ABLATION_DIR="$SYNCHRONAI_DIR/runs/fnirs_ablation_random"

if [ ! -f "$RANDOM_FEATURES/feature_index.csv" ]; then
    echo "ERROR: Random features not found at $RANDOM_FEATURES/feature_index.csv"
    echo "Run random feature extraction first."
    exit 1
fi

echo "=== Training ${ABLATION_MODEL_NAME}_lstm64 on random features ==="
python scripts/train_fnirs_from_features.py \
    --feature-dir "$RANDOM_FEATURES" \
    --save-dir "$ABLATION_DIR/${ABLATION_MODEL_NAME}_lstm64" \
    --label-column participant_type \
    --label-map "child:0,adult:1" \
    --hidden-dim 64 \
    --dropout 0.3 \
    --pool lstm \
    --learning-rate 3e-4 \
    --weight-decay 1e-2 \
    --warmup-epochs 3 \
    --patience 15 \
    --epochs 50 \
    --batch-size 32 \
    --num-workers 0 \
    --seed 42 \
    --include-tiers "gold,standard" \
    --holdout-tiers "gold,salvageable"

echo "=== Ablation complete ==="
ABLATION_EOF

echo "  Monitor: bjobs -g /\$USER/fnirs_ablation"

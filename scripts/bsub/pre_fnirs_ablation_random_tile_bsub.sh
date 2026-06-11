#!/bin/bash
SCRIPT_VERSION="pre_fnirs_ablation_random_tile_bsub-v1"
# =============================================================================
# fNIRS Ablation: Random Per-Pair Encoder — TILE re-run, Training Only
#
# Reads random tile features from:
#   $SYNCHRONAI_DIR/data/fnirs_perpair_{MODEL}_tile_random_features/
# Writes to:
#   $SYNCHRONAI_DIR/runs/fnirs_ablation_random_tile/{MODEL}_lstm64/
#
# Submit one size at a time:  PERPAIR_MODEL=large bash <this>
# Run AFTER: pre_fnirs_extract_random_tile_bsub.sh
# =============================================================================

MODEL_NAME="${PERPAIR_MODEL:-large}"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true
export ABLATION_MODEL_NAME="$MODEL_NAME"

DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

echo "=== [$SCRIPT_VERSION] ablation (tile): $MODEL_NAME (random encoder) ==="

ABL_SAVE_DIR="$SYNCHRONAI_DIR/runs/fnirs_ablation_random_tile/${MODEL_NAME}_lstm64"
if [ -f "$ABL_SAVE_DIR/best.pt" ] && [ -f "$ABL_SAVE_DIR/history.json" ]; then
    echo "  SKIP: ablation already trained — $ABL_SAVE_DIR/best.pt exists."
    exit 0
fi

case "$MODEL_NAME" in
    micro)  MEM_GB=8  ;;
    small)  MEM_GB=16 ;;
    medium) MEM_GB=24 ;;
    large)  MEM_GB=32 ;;
    *)      MEM_GB=16 ;;
esac
echo "  RAM: ${MEM_GB}GB"

bsub -J "synchronai-fnirs-random-tile-${MODEL_NAME}-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M $((MEM_GB * 1000000)) \
     -a 'docker(continuumio/anaconda3)' \
     -n 4 \
     -R "select[mem>${MEM_GB}GB] rusage[mem=${MEM_GB}GB] span[hosts=1]" \
     -oo "$LOG_DIR/fnirs_ablation_random_tile_${MODEL_NAME}_$DATE.log" \
     -g /$USER/fnirs_ablation \
     << 'ABLATION_EOF'
echo "=== ablation tile (random encoder) ==="
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONNOUSERSITE=1
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

RANDOM_FEATURES="$SYNCHRONAI_DIR/data/fnirs_perpair_${ABLATION_MODEL_NAME}_tile_random_features"
ABLATION_DIR="$SYNCHRONAI_DIR/runs/fnirs_ablation_random_tile"

if [ ! -f "$RANDOM_FEATURES/feature_index.csv" ]; then
    echo "ERROR: Random tile features not found at $RANDOM_FEATURES/feature_index.csv"
    echo "Run pre_fnirs_extract_random_tile_bsub.sh first."
    exit 1
fi

echo "=== Training ${ABLATION_MODEL_NAME}_lstm64 on random tile features ==="
"$ML_PY" scripts/train_fnirs_from_features.py \
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

echo "=== Ablation (tile) complete ==="
ABLATION_EOF

echo "  Monitor: bjobs -g /\$USER/fnirs_ablation"

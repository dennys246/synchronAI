#!/bin/bash
SCRIPT_VERSION="generative_fnirs_perpair_bsub-v2"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# =============================================================================
# fNIRS Per-Pair Generative Pretraining (single model)
#
# Reads config from environment variables set by pre_fnirs_perpair_pretrain_bsub.sh:
#   PERPAIR_NAME       — model name (small, medium, large)
#   PERPAIR_BASE_WIDTH — U-Net base width (16, 32, 64)
#   PERPAIR_EPOCHS     — max epochs (200)
#   PERPAIR_LR         — learning rate (1e-4)
# =============================================================================

NAME="${PERPAIR_NAME:-medium}"
BASE_WIDTH="${PERPAIR_BASE_WIDTH:-32}"
EPOCHS="${PERPAIR_EPOCHS:-200}"
LR="${PERPAIR_LR:-1e-4}"

echo "=== [$SCRIPT_VERSION] ==="
echo "=== fNIRS Per-Pair Pretraining: $NAME ==="
echo "  Base width: $BASE_WIDTH"
echo "  Epochs: $EPOCHS"
echo "  LR: $LR"

export SYNCHRONAI_DIR="${SYNCHRONAI_DIR:-/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI}"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
conda init
source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR

bash $SYNCHRONAI_DIR/scripts/generative_pretrain.sh \
    --save-dir "$SYNCHRONAI_DIR/runs/fnirs_perpair_${NAME}" \
    --unet-base-width "$BASE_WIDTH" \
    --duration-seconds 60 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lr-schedule cosine_restarts \
    --eval-gen-every 10 \
    --per-pair \
    --enable-qc \
    --sci-threshold 0.40 \
    --snr-threshold 2.0 \
    --cardiac-peak-ratio 2.0 \
    --no-require-cardiac

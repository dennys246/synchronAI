#!/bin/bash
SCRIPT_VERSION="generative_fnirs_perpair_micro_bsub-v2"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# =============================================================================
# fNIRS Per-Pair Generative Pretraining — Micro Model
#
# Smallest architecture: base_width=8, depth=2 (~15K params, 32-dim bottleneck)
# Tests whether a tiny encoder can learn HbO/HbR dynamics well enough for
# downstream classification. Designed for scalability: 200+ pairs per montage
# would still be cheap at 15K params per forward pass.
#
# Submit with:
#   export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"
#   export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
#   export LSF_DOCKER_PRESERVE_ENVIRONMENT=true
#   bsub -oo scripts/bsub/logs/fnirs_perpair_micro_$(date +'%m-%d').log \
#        < scripts/bsub/generative_fnirs_perpair_micro_bsub.sh
# =============================================================================

echo "=== [$SCRIPT_VERSION] ==="
echo "=== fNIRS Per-Pair Pretraining: micro (base=8, depth=2) ==="

export SYNCHRONAI_DIR="${SYNCHRONAI_DIR:-/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI}"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
conda init
source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR

bash $SYNCHRONAI_DIR/scripts/generative_pretrain.sh \
    --save-dir "$SYNCHRONAI_DIR/runs/fnirs_perpair_micro" \
    --unet-base-width 8 \
    --unet-depth 2 \
    --duration-seconds 60 \
    --epochs 200 \
    --lr 1e-4 \
    --lr-schedule cosine_restarts \
    --eval-gen-every 10 \
    --per-pair \
    --enable-qc \
    --sci-threshold 0.40 \
    --snr-threshold 2.0 \
    --cardiac-peak-ratio 2.0 \
    --no-require-cardiac

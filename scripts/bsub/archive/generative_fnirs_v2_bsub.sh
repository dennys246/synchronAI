#!/bin/bash
SCRIPT_VERSION="generative_fnirs_v2_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# =============================================================================
# fNIRS Diffusion v2 — Wider U-Net (64) + Cosine LR Restarts
#
# Changes from v1:
#   - base_width: 32 → 64 (4x more parameters)
#   - lr_schedule: constant → cosine_restarts (auto-decaying with warm restarts)
#   - Saves to runs/fnirs_diffusion_v2/ (separate from v1)
# =============================================================================

echo "=== [$SCRIPT_VERSION] ==="

conda init
source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR
pip install -e .

bash $SYNCHRONAI_DIR/scripts/generative_pretrain.sh \
  --duration-seconds 60 \
  --save-dir "$SYNCHRONAI_DIR/runs/fnirs_diffusion_v2" \
  --unet-base-width 64 \
  --lr-schedule cosine_restarts \
  --enable-qc --sci-threshold 0.75 --snr-threshold 5.0 --cardiac-peak-ratio 2.0

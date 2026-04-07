#!/bin/bash
SCRIPT_VERSION="generative_fnirs_v3_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# =============================================================================
# fNIRS Diffusion v3 — Subject-Grouped Val Split + FID/MMD Metrics
#
# Changes from v2:
#   - Validation split now groups by subject (prevents data leakage)
#   - FID and MMD generative quality metrics computed every 10 epochs
#   - Same architecture: base_width=64, cosine_restarts
#   - Saves to runs/fnirs_diffusion_v3/
# =============================================================================

echo "=== [$SCRIPT_VERSION] ==="

conda init
source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR
pip install -e .

bash $SYNCHRONAI_DIR/scripts/generative_pretrain.sh \
  --duration-seconds 60 \
  --save-dir "$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3" \
  --unet-base-width 64 \
  --lr-schedule cosine_restarts \
  --eval-gen-every 10 \
  --enable-qc --sci-threshold 0.75 --snr-threshold 5.0 --cardiac-peak-ratio 2.0

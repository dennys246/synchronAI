#!/bin/sh
SCRIPT_VERSION="pre_generative_fnirs_v3_bsub-v1"
# =============================================================================
# Submit fNIRS Diffusion v3 Training
#
# Subject-grouped validation split + FID/MMD generative quality metrics.
# Same architecture as v2 (base_width=64, cosine restarts).
# Saves to runs/fnirs_diffusion_v3/ — does NOT touch v1 or v2 runs.
# =============================================================================

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH='/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:/storage1/fs1/perlmansusan/Active/moochie/github/synchronyAI':$PYTHONPATH

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"

export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

# Weights & Biases — set your API key from https://wandb.ai/authorize
# Or use WANDB_MODE=offline to log locally and sync later with: wandb sync ./wandb/offline-run-*
export WANDB_API_KEY="${WANDB_API_KEY:-}"

export DATE=$(date +'%m-%d')

echo "=== [$SCRIPT_VERSION] ==="

bsub -J synchronai-fnirs-gen-v3-$DATE -oo $SYNCHRONAI_DIR/scripts/bsub/logs/synchronai_fnirs_gen_v3_$DATE.log -g /$USER/preprocessing < /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/generative_fnirs_v3_bsub.sh

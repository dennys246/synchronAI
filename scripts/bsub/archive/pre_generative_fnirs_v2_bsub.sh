#!/bin/sh
SCRIPT_VERSION="pre_generative_fnirs_v2_bsub-v1"
# =============================================================================
# Submit fNIRS Diffusion v2 Training
#
# Wider U-Net (base_width=64) + cosine LR decay with warm restarts.
# Saves to runs/fnirs_diffusion_v2/ — does NOT touch the current v1 run.
# =============================================================================

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH='/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:/storage1/fs1/perlmansusan/Active/moochie/github/synchronyAI':$PYTHONPATH

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"

export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')

echo "=== [$SCRIPT_VERSION] ==="

bsub -J synchronai-fnirs-gen-v2-$DATE -oo $SYNCHRONAI_DIR/scripts/bsub/logs/synchronai_fnirs_gen_v2_$DATE.log -g /$USER/preprocessing < /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/generative_fnirs_v2_bsub.sh

#!/bin/sh
SCRIPT_VERSION="pre_multimodal_synchrony_bsub-v1"

# Pre-submission script for multi-modal synchrony classifier training
# Sets up environment and submits bsub job to LSF cluster

# Export conda directories
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

# Set synchronAI project directory
export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

# Set up PATH
export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH='/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:/storage1/fs1/perlmansusan/Active/moochie/github/synchronyAI':$PYTHONPATH

# Docker volume mounts
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"

# Preserve environment in Docker
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

# Generate date stamp for job naming
export DATE=$(date +'%m-%d')

echo "=== [$SCRIPT_VERSION] ==="

# Submit job to LSF with bsub
bsub \
    -J synchronai-multimodal-synchrony-$DATE \
    -oo $SYNCHRONAI_DIR/scripts/bsub/logs/synchronai_multimodal_$DATE.log \
    -g /$USER/preprocessing \
    < /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/multimodal_synchrony_bsub.sh

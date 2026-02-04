#!/bin/sh

# export OUTPUT_DIRS
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH='/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:/storage1/fs1/perlmansusan/Active/moochie/github/synchronyAI':$PYTHONPATH

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"

export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')

bsub -J synchronai-fnirs-generative-$DATE -oo $SYNCHRONAI_DIR/scripts/bsub/logs/synchronai_fnirs_gen_$DATE.log -g /$USER/preprocessing < /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/generative_fnirs_bsub.sh

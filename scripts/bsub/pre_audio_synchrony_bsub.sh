#!/bin/sh
SCRIPT_VERSION="pre_audio_synchrony_bsub-v1"

# export OUTPUT_DIRS
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH='/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI':$PYTHONPATH

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"

export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')

echo "=== [$SCRIPT_VERSION] ==="

bsub -J synchronai-audio-$DATE \
     -oo $SYNCHRONAI_DIR/scripts/bsub/logs/synchronai_audio_$DATE.log \
     -g /$USER/preprocessing \
     < $SYNCHRONAI_DIR/scripts/bsub/audio_synchrony_bsub.sh

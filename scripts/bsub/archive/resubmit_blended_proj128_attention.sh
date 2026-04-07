#!/bin/sh
SCRIPT_VERSION="resubmit_blended_proj128_attention-v1"
# One-off resubmit for blended_proj128_attention (accidentally canceled)

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"
export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"
export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
SWEEP_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/wavlm_audio_sweep_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"

export SWEEP_ENCODER="wavlm-base-plus"
export SWEEP_OUTPUT_BASE="runs/audio_sweep_v2"
export SWEEP_FEATURE_DIR="data/wavlm_baseplus_features"

export SWEEP_RUN_NAME="blended_proj128_attention"
export SWEEP_TEMPORAL_AGG="attention"
export SWEEP_PROJECT_DIM="128"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"
export SWEEP_PATIENCE="15"

echo "=== [$SCRIPT_VERSION] ==="
echo "Resubmitting blended_proj128_attention..."
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

echo "Done. Monitor with: bjobs -g /$USER/audio_sweep_v2"

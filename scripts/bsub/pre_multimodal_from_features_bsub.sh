#!/bin/sh
SCRIPT_VERSION="pre_multimodal_from_features_bsub-v2"
# =============================================================================
# Submit CPU-only multi-modal training on pre-extracted features.
#
# Joins DINOv2 video features + WavLM-base-plus audio features by
# (video_path, second), trains a small per-modality LSTM + concat head.
# Designed for environments without GPU access — all backbones are
# pre-computed, so per-batch cost is just two small LSTMs.
# =============================================================================

# Shared cluster environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
TRAIN_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/multimodal_from_features_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

# Run config (overridable via environment variables; see multimodal_from_features_bsub.sh)
export MM_VIDEO_FEATURE_DIR="${MM_VIDEO_FEATURE_DIR:-data/dinov2_features_meanpatch}"
export MM_AUDIO_FEATURE_DIR="${MM_AUDIO_FEATURE_DIR:-data/wavlm_baseplus_features}"
export MM_SAVE_DIR="${MM_SAVE_DIR:-runs/multimodal_features/v2_baseline}"

echo "=== [$SCRIPT_VERSION] ==="
echo "Submitting multi-modal feature training job..."
echo "  Arch:           ${MM_ARCH:-v2}"
echo "  Video features: $MM_VIDEO_FEATURE_DIR"
echo "  Audio features: $MM_AUDIO_FEATURE_DIR"
echo "  Save dir:       $MM_SAVE_DIR"

# %J in -oo expands to the LSF job ID, so concurrent same-day sweep submissions
# don't overwrite each other's logs.
bsub \
    -J synchronai-multimodal-features-$DATE \
    -oo "$LOG_DIR/synchronai_multimodal_features_${DATE}_%J.log" \
    -g /$USER/multimodal \
    < "$TRAIN_SCRIPT"

echo "Submitted. Tail the log at:"
echo "  $LOG_DIR/synchronai_multimodal_features_${DATE}_<JOBID>.log"

#!/bin/sh
SCRIPT_VERSION="pre_perperson_video_features_bsub-v1"
# =============================================================================
# Submit the per-person video feature construction job. Mirrors the
# pre_pose_extract_bsub.sh / pre_dinov2_patch_extract_bsub.sh launcher
# pattern.
#
# Usage:
#   bash scripts/bsub/pre_perperson_video_features_bsub.sh
#
# Smoke test (first 100 segments, ~1 min):
#   PP_LIMIT=100 bash scripts/bsub/pre_perperson_video_features_bsub.sh
# =============================================================================

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
EXTRACT_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/perperson_video_features_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

export PP_PATCH_FEATURE_DIR="${PP_PATCH_FEATURE_DIR:-data/dinov2_features_patches}"
export PP_POSE_FEATURE_DIR="${PP_POSE_FEATURE_DIR:-data/pose_features}"
export PP_OUTPUT_DIR="${PP_OUTPUT_DIR:-data/perperson_video_features}"

echo "=== [$SCRIPT_VERSION] ==="
echo "Submitting per-person video feature construction..."
echo "  Patch dir:   $PP_PATCH_FEATURE_DIR"
echo "  Pose dir:    $PP_POSE_FEATURE_DIR"
echo "  Output dir:  $PP_OUTPUT_DIR"
if [ -n "${PP_LIMIT:-}" ]; then
    echo "  LIMIT:       $PP_LIMIT"
fi

bsub \
    -J synchronai-perperson-build-$DATE \
    -oo "$LOG_DIR/synchronai_perperson_build_${DATE}_%J.log" \
    -g /$USER/preprocessing \
    < "$EXTRACT_SCRIPT"

echo ""
echo "Submitted. Tail the log at:"
echo "  $LOG_DIR/synchronai_perperson_build_${DATE}_<JOBID>.log"
echo ""
echo "Once complete, train the cross-person probe:"
echo "  bash scripts/bsub/submit_crossperson_probe_fold0.sh"

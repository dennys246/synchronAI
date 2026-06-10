#!/bin/sh
SCRIPT_VERSION="pre_dinov2_patch_extract_bsub-v1"
# =============================================================================
# Submit DINOv2 patch-grid feature extraction.
# Mirrors pre_pose_extract_bsub.sh: this script sets the cluster env vars +
# log paths and submits dinov2_patch_extract_bsub.sh via bsub.
#
# Usage:
#   bash scripts/bsub/pre_dinov2_patch_extract_bsub.sh
#
# With smoke test (just first N rows of labels.csv, ~5 min):
#   PATCH_LABELS_FILE=data/labels.csv  # default
#   # Smoke testing requires editing extract_dinov2_patch_features.py to add
#   # --limit; not currently wired. For now, kill the job after a few
#   # minutes if you just want to validate output format.
# =============================================================================

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
EXTRACT_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/dinov2_patch_extract_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

export PATCH_LABELS_FILE="${PATCH_LABELS_FILE:-data/labels.csv}"
export PATCH_OUTPUT_DIR="${PATCH_OUTPUT_DIR:-data/dinov2_features_patches}"
export PATCH_BACKBONE="${PATCH_BACKBONE:-dinov2-base}"

echo "=== [$SCRIPT_VERSION] ==="
echo "Submitting DINOv2 patch extraction job..."
echo "  Labels:        $PATCH_LABELS_FILE"
echo "  Output dir:    $PATCH_OUTPUT_DIR"
echo "  Backbone:      $PATCH_BACKBONE"

bsub \
    -J synchronai-dinov2-patch-extract-$DATE \
    -oo "$LOG_DIR/synchronai_dinov2_patch_extract_${DATE}_%J.log" \
    -g /$USER/preprocessing \
    < "$EXTRACT_SCRIPT"

echo ""
echo "Submitted. Tail the log at:"
echo "  $LOG_DIR/synchronai_dinov2_patch_extract_${DATE}_<JOBID>.log"
echo ""
echo "Once complete, kick off probe 1 (which now reads the new dir):"
echo "  bash scripts/bsub/submit_patch_probe_fold0.sh"

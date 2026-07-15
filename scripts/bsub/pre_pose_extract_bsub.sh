#!/bin/sh
SCRIPT_VERSION="pre_pose_extract_bsub-v1"
# =============================================================================
# Submit pose-keypoint extraction (probe 3). Mirrors the
# pre_multimodal_from_features_bsub.sh launcher pattern: this script sets
# the cluster env vars + log paths and submits pose_extract_bsub.sh via bsub.
#
# Usage:
#   bash scripts/bsub/pre_pose_extract_bsub.sh
#
# Or with overrides (smoke test):
#   POSE_LIMIT=100 bash scripts/bsub/pre_pose_extract_bsub.sh
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
EXTRACT_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/pose_extract_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

# Run config (overridable via environment variables; see pose_extract_bsub.sh)
export POSE_LABELS_FILE="${POSE_LABELS_FILE:-data/labels.csv}"
export POSE_OUTPUT_DIR="${POSE_OUTPUT_DIR:-data/pose_features}"
export POSE_MODEL_WEIGHTS="${POSE_MODEL_WEIGHTS:-scripts/yolo26n-pose.pt}"

echo "=== [$SCRIPT_VERSION] ==="
echo "Submitting pose extraction job..."
echo "  Labels:        $POSE_LABELS_FILE"
echo "  Output dir:    $POSE_OUTPUT_DIR"
echo "  Model:         $POSE_MODEL_WEIGHTS"
if [ -n "${POSE_LIMIT:-}" ]; then
    echo "  LIMIT:         $POSE_LIMIT (smoke-test mode)"
fi

bsub \
    -J synchronai-pose-extract-$DATE \
    -oo "$LOG_DIR/synchronai_pose_extract_${DATE}_%J.log" \
    -g /$USER/pose \
    < "$EXTRACT_SCRIPT"

echo ""
echo "Submitted. Tail the log at:"
echo "  $LOG_DIR/synchronai_pose_extract_${DATE}_<JOBID>.log"
echo ""
echo "Once complete, train a pose-only or pose+video+audio model by passing"
echo "  --video-feature-dir data/pose_features  (or fuse via merge_feature_indices)"
echo "Note: the existing trainer's collate expects 2D video features. Pose is 4D"
echo "(12, 2, 17, 3) → flatten to (12, 102) when wiring up training, or write"
echo "a small MultiModalV2Pose variant that keeps the (2, 17, 3) structure."

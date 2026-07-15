#!/bin/sh
SCRIPT_VERSION="pre_compute_irr_bsub-v1"
# =============================================================================
# Submit IRR computation job.
#
# Usage:
#   IRR_LABEL_DIR=/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/label_data \
#     bash scripts/bsub/pre_compute_irr_bsub.sh
#
# Or just bash it after exporting the env var. Most arguments are
# overridable via env (see compute_irr_bsub.sh).
# =============================================================================

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
EXTRACT_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/compute_irr_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

if [ -z "$IRR_LABEL_DIR" ]; then
    echo "ERROR: IRR_LABEL_DIR must be set."
    echo "Example:"
    echo "  IRR_LABEL_DIR=/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/label_data \\"
    echo "    bash scripts/bsub/pre_compute_irr_bsub.sh"
    exit 1
fi

echo "=== [$SCRIPT_VERSION] ==="
echo "Submitting IRR computation..."
echo "  Label dir:  $IRR_LABEL_DIR"
echo "  Output dir: ${IRR_OUTPUT_DIR:-runs/irr_analysis}"

bsub \
    -J synchronai-compute-irr-$DATE \
    -oo "$LOG_DIR/synchronai_compute_irr_${DATE}_%J.log" \
    -g /$USER/irr \
    < "$EXTRACT_SCRIPT"

echo ""
echo "Submitted. Tail the log at:"
echo "  $LOG_DIR/synchronai_compute_irr_${DATE}_<JOBID>.log"
echo ""
echo "Once complete:"
echo "  cat ${IRR_OUTPUT_DIR:-runs/irr_analysis}/irr_summary.txt | head -40"

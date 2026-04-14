#!/bin/sh
SCRIPT_VERSION="pre_multirez_bsub-v1"
# =============================================================================
# DINOv2 Progressive Growing Pipeline — Stage Launcher
#
# Usage:
#   bash scripts/bsub/pre_multirez_bsub.sh <stage>
#
# Stages:
#   1  Extract multi-resolution features (112×112, 168×168)
#      ~59K windows × 2 resolutions. CPU-heavy, requires opencv + transformers.
#
#   2  Submit progressive sweep (6 ablation training jobs)
#      Requires stage 1 features to exist. CPU-only training.
#
#   3  Temperature-scale the best checkpoint from the sweep
#      Run after sweep completes. Pass best run name as 2nd arg:
#        bash pre_multirez_bsub.sh 3 prog_full
#
# Examples:
#   bash scripts/bsub/pre_multirez_bsub.sh 1        # extract features
#   bash scripts/bsub/pre_multirez_bsub.sh 2        # submit sweep
#   bash scripts/bsub/pre_multirez_bsub.sh 3 prog_full  # calibrate best run
#
# Monitor:
#   bjobs -g /$USER/prog_sweep
#   bjobs -g /$USER/multirez
# =============================================================================

set -e

STAGE="${1:-}"
BEST_RUN="${2:-prog_full}"

if [ -z "$STAGE" ]; then
    echo "=== [$SCRIPT_VERSION] ==="
    echo "Usage: bash scripts/bsub/pre_multirez_bsub.sh <stage> [best_run_name]"
    echo ""
    echo "Stages:"
    echo "  1  Extract multi-resolution features (112, 168)"
    echo "  2  Submit progressive sweep (6 ablation jobs)"
    echo "  3  Temperature-scale best checkpoint"
    echo ""
    echo "Examples:"
    echo "  bash scripts/bsub/pre_multirez_bsub.sh 1"
    echo "  bash scripts/bsub/pre_multirez_bsub.sh 2"
    echo "  bash scripts/bsub/pre_multirez_bsub.sh 3 prog_full"
    exit 1
fi

# =============================================================================
# Shared environment
# =============================================================================

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$SYNCHRONAI_DIR:$SYNCHRONAI_DIR/scripts:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

# Feature directories
FEAT_112="data/dinov2_features_small_112"
FEAT_168="data/dinov2_features_small_168"
FEAT_224="data/dinov2_features_small_meanpatch"

# =============================================================================
# Stage 1: Extract multi-resolution features
# =============================================================================

if [ "$STAGE" = "1" ]; then
    echo "=========================================="
    echo "  Stage 1: Multi-Resolution Feature Extraction"
    echo "  Date: $DATE"
    echo "=========================================="
    echo ""
    echo "  Submitting extraction job (112×112 + 168×168)..."
    echo "  224×224 features already exist at: ${FEAT_224}"
    echo ""

    EXTRACT_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/dinov2_extract_multirez_bsub.sh"

    bsub -J "synchronai-multirez-extract-$DATE" \
         -oo "$LOG_DIR/multirez_extract_$DATE.log" \
         -g /$USER/multirez \
         < "$EXTRACT_SCRIPT"

    echo "  Job submitted."
    echo "  Monitor with: bjobs -g /$USER/multirez"
    echo ""
    echo "  After completion, verify features exist:"
    echo "    ls ${FEAT_112}/feature_index.csv"
    echo "    ls ${FEAT_168}/feature_index.csv"
    echo ""
    echo "  Then run: bash scripts/bsub/pre_multirez_bsub.sh 2"

# =============================================================================
# Stage 2: Submit progressive sweep
# =============================================================================

elif [ "$STAGE" = "2" ]; then
    echo "=========================================="
    echo "  Stage 2: Progressive Growing Sweep"
    echo "  Date: $DATE"
    echo "=========================================="

    # Verify prerequisite features exist
    MISSING=0
    for FDIR in "$SYNCHRONAI_DIR/$FEAT_112" "$SYNCHRONAI_DIR/$FEAT_168" "$SYNCHRONAI_DIR/$FEAT_224"; do
        if [ ! -f "${FDIR}/feature_index.csv" ]; then
            echo "  ERROR: Missing features at ${FDIR}"
            MISSING=1
        fi
    done

    if [ "$MISSING" = "1" ]; then
        echo ""
        echo "  Run stage 1 first: bash scripts/bsub/pre_multirez_bsub.sh 1"
        exit 1
    fi

    echo "  All feature directories verified."
    echo ""

    SWEEP_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/dinov2_progressive_sweep_bsub.sh"

    # Shared config (best from previous sweep: small_heavy_reg)
    export SWEEP_TEMPORAL_AGG="lstm"
    export SWEEP_HIDDEN_DIM="128"
    export SWEEP_DROPOUT="0.7"
    export SWEEP_LEARNING_RATE="3e-5"
    export SWEEP_WEIGHT_DECAY="1e-2"
    export SWEEP_PATIENCE="15"
    export SWEEP_LR_RESTART_PERIOD="10"

    echo "  Submitting 6 ablation training jobs..."
    echo ""

    # --- Run 1: flat_baseline ---
    export SWEEP_RUN_NAME="flat_baseline"
    export SWEEP_MODE="flat"
    export SWEEP_MIXUP_ALPHA="0"
    export SWEEP_LR_SCHEDULE="cosine"

    echo "  1/6: ${SWEEP_RUN_NAME} — reproduce previous best (control)"
    bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
         -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
         -g /$USER/prog_sweep \
         < "$SWEEP_SCRIPT"

    # --- Run 2: flat_mixup ---
    export SWEEP_RUN_NAME="flat_mixup"
    export SWEEP_MODE="flat"
    export SWEEP_MIXUP_ALPHA="0.2"
    export SWEEP_LR_SCHEDULE="cosine"

    echo "  2/6: ${SWEEP_RUN_NAME} — flat 224 + mixup(0.2)"
    bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
         -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
         -g /$USER/prog_sweep \
         < "$SWEEP_SCRIPT"

    # --- Run 3: flat_restarts ---
    export SWEEP_RUN_NAME="flat_restarts"
    export SWEEP_MODE="flat"
    export SWEEP_MIXUP_ALPHA="0"
    export SWEEP_LR_SCHEDULE="cosine_restarts"

    echo "  3/6: ${SWEEP_RUN_NAME} — flat 224 + cosine restarts"
    bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
         -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
         -g /$USER/prog_sweep \
         < "$SWEEP_SCRIPT"

    # --- Run 4: flat_mixup_restarts ---
    export SWEEP_RUN_NAME="flat_mixup_restarts"
    export SWEEP_MODE="flat"
    export SWEEP_MIXUP_ALPHA="0.2"
    export SWEEP_LR_SCHEDULE="cosine_restarts"

    echo "  4/6: ${SWEEP_RUN_NAME} — flat 224 + mixup + restarts"
    bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
         -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
         -g /$USER/prog_sweep \
         < "$SWEEP_SCRIPT"

    # --- Run 5: prog_baseline ---
    export SWEEP_RUN_NAME="prog_baseline"
    export SWEEP_MODE="progressive"
    export SWEEP_MIXUP_ALPHA="0"
    export SWEEP_LR_SCHEDULE="cosine"

    echo "  5/6: ${SWEEP_RUN_NAME} — progressive (112→168→224)"
    bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
         -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
         -g /$USER/prog_sweep \
         < "$SWEEP_SCRIPT"

    # --- Run 6: prog_full ---
    export SWEEP_RUN_NAME="prog_full"
    export SWEEP_MODE="progressive"
    export SWEEP_MIXUP_ALPHA="0.2"
    export SWEEP_LR_SCHEDULE="cosine_restarts"

    echo "  6/6: ${SWEEP_RUN_NAME} — progressive + mixup + restarts (full)"
    bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
         -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
         -g /$USER/prog_sweep \
         < "$SWEEP_SCRIPT"

    echo ""
    echo "=========================================="
    echo "  6 training jobs submitted"
    echo ""
    echo "  Monitor: bjobs -g /$USER/prog_sweep"
    echo ""
    echo "  Ablation:"
    echo "    1. flat_baseline        — control (previous best config)"
    echo "    2. flat_mixup           — +mixup only"
    echo "    3. flat_restarts        — +cosine restarts only"
    echo "    4. flat_mixup_restarts  — +mixup +restarts"
    echo "    5. prog_baseline        — +progressive resolution only"
    echo "    6. prog_full            — +progressive +mixup +restarts"
    echo ""
    echo "  After all complete:"
    echo "    python scripts/compare_sweep_results.py --sweep-dir runs/dinov2_progressive"
    echo "    bash scripts/bsub/pre_multirez_bsub.sh 3 <best_run_name>"
    echo "=========================================="

# =============================================================================
# Stage 3: Temperature scaling on best checkpoint
# =============================================================================

elif [ "$STAGE" = "3" ]; then
    echo "=========================================="
    echo "  Stage 3: Temperature Scaling"
    echo "  Best run: ${BEST_RUN}"
    echo "  Date: $DATE"
    echo "=========================================="

    CHECKPOINT="$SYNCHRONAI_DIR/runs/dinov2_progressive/${BEST_RUN}/best.pt"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "  ERROR: Checkpoint not found: ${CHECKPOINT}"
        echo "  Available runs:"
        ls -d "$SYNCHRONAI_DIR/runs/dinov2_progressive"/*/ 2>/dev/null | while read d; do
            basename "$d"
        done
        exit 1
    fi

    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  Feature dir: ${FEAT_224}"
    echo ""

    CALIBRATE_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/dinov2_calibrate_bsub.sh"

    # Create a lightweight calibration bsub job inline
    cat > "/tmp/calibrate_${BEST_RUN}.sh" << 'CALIBRATE_EOF'
#!/bin/bash
SCRIPT_VERSION="pre_multirez_bsub-v1"
echo "=== [pre_multirez_bsub-v1] ==="
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 16000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>16GB] rusage[mem=16GB]'

source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR:$SYNCHRONAI_DIR/scripts:$PYTHONPATH"

python scripts/calibrate_temperature.py \
    --checkpoint "${CALIBRATE_CHECKPOINT}" \
    --feature-dir "${CALIBRATE_FEAT_DIR}"

echo "Calibration complete. Check:"
echo "  ${CALIBRATE_CHECKPOINT%best.pt}calibrated.pt"
echo "  ${CALIBRATE_CHECKPOINT%best.pt}calibration_report.json"
CALIBRATE_EOF

    export CALIBRATE_CHECKPOINT="$CHECKPOINT"
    export CALIBRATE_FEAT_DIR="$FEAT_224"

    echo "  Submitting calibration job..."
    bsub -J "synchronai-calibrate-${BEST_RUN}-$DATE" \
         -oo "$LOG_DIR/calibrate_${BEST_RUN}_$DATE.log" \
         -g /$USER/multirez \
         < "/tmp/calibrate_${BEST_RUN}.sh"

    echo "  Job submitted."
    echo "  Monitor with: bjobs -g /$USER/multirez"
    echo ""
    echo "  Output will be at:"
    echo "    runs/dinov2_progressive/${BEST_RUN}/calibrated.pt"
    echo "    runs/dinov2_progressive/${BEST_RUN}/calibration_report.json"

# =============================================================================
# Invalid stage
# =============================================================================

else
    echo "ERROR: Unknown stage '${STAGE}'"
    echo ""
    echo "Valid stages: 1, 2, 3"
    echo "  1  Extract multi-resolution features"
    echo "  2  Submit progressive sweep"
    echo "  3  Temperature-scale best checkpoint"
    exit 1
fi
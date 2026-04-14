#!/bin/sh
SCRIPT_VERSION="pre_dinov2_progressive_sweep_bsub-v1"
# =============================================================================
# Submit DINOv2 Progressive Growing Sweep
#
# Ablation study comparing progressive resolution training against flat
# baselines with new regularization techniques (mixup, cosine restarts).
#
# Prerequisite: multi-resolution features must be extracted first.
# Run dinov2_extract_multirez_bsub.sh BEFORE this script.
#
# All runs use dinov2-small (384-dim), LSTM temporal aggregation.
# Best config from previous sweep: dropout=0.7, wd=1e-2, lr=3e-5.
#
# After all jobs complete, run:
#   python scripts/compare_sweep_results.py --sweep-dir runs/dinov2_progressive
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

# Weights & Biases — set your API key from https://wandb.ai/authorize
# Or use WANDB_MODE=offline to log locally and sync later with: wandb sync ./wandb/offline-run-*
export WANDB_API_KEY="${WANDB_API_KEY:-}"

export DATE=$(date +'%m-%d')
SWEEP_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/dinov2_progressive_sweep_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

echo "=== [$SCRIPT_VERSION] ==="
echo "=========================================="
echo "  DINOv2 Progressive Growing Sweep"
echo "  Ablation: progressive vs flat + new regularization"
echo "  Date: $DATE"
echo "=========================================="
echo ""
echo "Submitting 6 training jobs..."

# --- Run 1: flat_baseline — reproduces small_heavy_reg (previous best) ---
# No mixup, no restarts — pure baseline for comparison
export SWEEP_RUN_NAME="flat_baseline"
export SWEEP_MODE="flat"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_MIXUP_ALPHA="0"
export SWEEP_LR_SCHEDULE="cosine"
export SWEEP_LR_RESTART_PERIOD="10"

echo "  1/6: ${SWEEP_RUN_NAME} — reproduce previous best (no new features)"
bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/prog_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 2: flat_mixup — add feature mixup only ---
export SWEEP_RUN_NAME="flat_mixup"
export SWEEP_MODE="flat"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_MIXUP_ALPHA="0.2"
export SWEEP_LR_SCHEDULE="cosine"
export SWEEP_LR_RESTART_PERIOD="10"

echo "  2/6: ${SWEEP_RUN_NAME} — flat 224 + mixup(0.2)"
bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/prog_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 3: flat_restarts — add cosine restarts only ---
export SWEEP_RUN_NAME="flat_restarts"
export SWEEP_MODE="flat"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_MIXUP_ALPHA="0"
export SWEEP_LR_SCHEDULE="cosine_restarts"
export SWEEP_LR_RESTART_PERIOD="10"

echo "  3/6: ${SWEEP_RUN_NAME} — flat 224 + cosine restarts (T_0=10)"
bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/prog_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 4: flat_mixup_restarts — both mixup + restarts (flat) ---
export SWEEP_RUN_NAME="flat_mixup_restarts"
export SWEEP_MODE="flat"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_MIXUP_ALPHA="0.2"
export SWEEP_LR_SCHEDULE="cosine_restarts"
export SWEEP_LR_RESTART_PERIOD="10"

echo "  4/6: ${SWEEP_RUN_NAME} — flat 224 + mixup(0.2) + restarts"
bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/prog_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 5: prog_baseline — progressive resolution (112→168→224) ---
export SWEEP_RUN_NAME="prog_baseline"
export SWEEP_MODE="progressive"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_MIXUP_ALPHA="0"
export SWEEP_LR_SCHEDULE="cosine"
export SWEEP_LR_RESTART_PERIOD="10"

echo "  5/6: ${SWEEP_RUN_NAME} — progressive (112→168→224), no mixup"
bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/prog_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 6: prog_full — progressive + mixup + restarts (the full combo) ---
export SWEEP_RUN_NAME="prog_full"
export SWEEP_MODE="progressive"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="3e-5"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_MIXUP_ALPHA="0.2"
export SWEEP_LR_SCHEDULE="cosine_restarts"
export SWEEP_LR_RESTART_PERIOD="10"

echo "  6/6: ${SWEEP_RUN_NAME} — progressive + mixup(0.2) + restarts (full)"
bsub -J "synchronai-prog-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/prog_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/prog_sweep \
     < "$SWEEP_SCRIPT"

echo ""
echo "=========================================="
echo "  6 training jobs submitted"
echo ""
echo "  Monitor with: bjobs -g /$USER/prog_sweep"
echo ""
echo "  Ablation structure:"
echo "    1. flat_baseline        — previous best (control)"
echo "    2. flat_mixup           — +mixup only"
echo "    3. flat_restarts        — +cosine restarts only"
echo "    4. flat_mixup_restarts  — +mixup +restarts"
echo "    5. prog_baseline        — +progressive resolution only"
echo "    6. prog_full            — +progressive +mixup +restarts"
echo ""
echo "  After all complete:"
echo "    python scripts/compare_sweep_results.py --sweep-dir runs/dinov2_progressive"
echo "=========================================="
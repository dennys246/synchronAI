#!/bin/sh
SCRIPT_VERSION="pre_fnirs_perpair_pretrain_bsub-v1"
# =============================================================================
# fNIRS Per-Pair Generative Pretraining Sweep
#
# Trains 3 DDPM models on individual source-detector pairs (feature_dim=2).
# Each 60s window from a recording produces 10 samples (one per pair),
# yielding ~670K training samples from ~67K windows.
#
# The per-pair model learns universal HbO/HbR hemodynamic dynamics that
# generalize to any montage configuration (any number of source-detector pairs).
#
# Quality control pipeline (enabled by default):
#   - SCI >= 0.75: Scalp coupling index (cardiac-band correlation between
#     wavelengths). Rejects poorly-coupled channels.
#   - SNR >= 5.0: PSD-based signal-to-noise ratio. Signal = 0.01-0.2 Hz
#     (HRF band), noise = outside that range. Rejects noisy scans.
#   - Cardiac band (0.8-1.5 Hz): Verifies heartbeat is detectable in raw
#     optical data. Channels without cardiac signal are excluded.
#
# Sweep variants:
#   Micro:  base_width=8,  depth=2  (~15K params, 32-dim bottleneck)
#   Small:  base_width=16, depth=3  (~100K params)
#   Medium: base_width=32, depth=3  (~800K params)
#   Large:  base_width=64, depth=3  (~3.5M params, same as 20-channel v3)
#
# After pretraining completes:
#   1. Convert best model's weights to PyTorch
#   2. Run child/adult classification sweep
#   3. Compare against 20-channel baseline (AUC 0.974)
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"

export DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

PRETRAIN_SCRIPT="$SYNCHRONAI_DIR/scripts/generative_pretrain.sh"

echo "=== [$SCRIPT_VERSION] ==="
echo "=========================================="
echo "  fNIRS Per-Pair Generative Pretraining"
echo "  4 architecture sweep"
echo "  Date: $DATE"
echo "=========================================="

# Helper function — uses exported env vars that LSF passes through
submit_pretrain() {
    local NAME="$1"
    local BASE_WIDTH="$2"
    local DEPTH="$3"
    local EPOCHS="$4"
    local LR="$5"

    echo "  $NAME (base=$BASE_WIDTH, depth=$DEPTH, epochs=$EPOCHS)"

    export PERPAIR_NAME="$NAME"
    export PERPAIR_BASE_WIDTH="$BASE_WIDTH"
    export PERPAIR_EPOCHS="$EPOCHS"
    export PERPAIR_LR="$LR"

    bsub -J "synchronai-fnirs-perpair-${NAME}-$DATE" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 99000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 40 \
         -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]' \
         -oo "$LOG_DIR/fnirs_perpair_${NAME}_$DATE.log" \
         -g /$USER/fnirs_perpair \
         < "$SYNCHRONAI_DIR/scripts/bsub/generative_fnirs_perpair_bsub.sh"
}

echo ""
echo "Submitting 4 pretraining jobs..."

# --- Micro: base=8, depth=2 (~15K params, 32-dim bottleneck) ---
# Smallest possible. Tests if a tiny encoder suffices for HbO/HbR dynamics.
# Uses its own bsub script because depth=2 differs from the default depth=3.
echo "  micro (base=8, depth=2, epochs=200)"
bsub -J "synchronai-fnirs-perpair-micro-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 99000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 40 \
     -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]' \
     -oo "$LOG_DIR/fnirs_perpair_micro_$DATE.log" \
     -g /$USER/fnirs_perpair \
     < "$SYNCHRONAI_DIR/scripts/bsub/generative_fnirs_perpair_micro_bsub.sh"

# Remaining models use depth=3 (default). Sweep is over width only.

# --- Small: base=16, depth=3 (~100K params) ---
# Narrowest model. Tests if 2-channel HbO/HbR dynamics need minimal capacity.
submit_pretrain "small"  16 3 200 "1e-4"

# --- Medium: base=32, depth=3 (~800K params) ---
# Middle ground. Half the width of the 20-channel v3 model.
submit_pretrain "medium" 32 3 200 "1e-4"

# --- Large: base=64, depth=3 (~3.5M params) ---
# Same architecture as the 20-channel v3 model.
submit_pretrain "large"  64 3 200 "1e-4"

echo ""
echo "=========================================="
echo "  4 pretraining jobs submitted"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_perpair"
echo ""
echo "  After all complete, run the transfer learning sweep:"
echo "    sh scripts/bsub/pre_fnirs_perpair_transfer_bsub.sh"
echo "=========================================="

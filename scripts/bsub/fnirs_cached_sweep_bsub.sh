#!/bin/bash
#BSUB -J fnirs_cached_sweep
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 32000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 16
#BSUB -R 'select[mem>32GB && tmp>16GB] rusage[mem=32GB, tmp=16GB] span[hosts=1]'
# #BSUB directives MUST stay directly under the shebang (before any command),
# or LSF ignores them and uses the whole script as the job name.

SCRIPT_VERSION="fnirs_cached_sweep_bsub-v1"

# =============================================================================
# fNIRS per-pair generative training FROM the packed window cache (no per-epoch
# preprocessing). One model size per job; submit 4 in parallel for the sweep.
#
# Env config (override per size):
#   SWEEP_NAME        model name (micro/small/medium/large)
#   SWEEP_BASE_WIDTH  U-Net base width (8/16/32/64)
#   SWEEP_DEPTH       U-Net depth (2 for micro, else 3)
#   SWEEP_EPOCHS      epochs (default 30, from the probe's val-loss plateau)
#   SWEEP_BATCH       batch size (default 64 — good CPU throughput)
#   SWEEP_CACHE_DIR   packed window cache dir
#   SWEEP_SAVE_DIR    output run dir (default runs/fnirs_perpair_${NAME}_tile)
# =============================================================================

NAME="${SWEEP_NAME:-small}"
BASE_WIDTH="${SWEEP_BASE_WIDTH:-16}"
DEPTH="${SWEEP_DEPTH:-3}"
EPOCHS="${SWEEP_EPOCHS:-30}"
BATCH="${SWEEP_BATCH:-64}"
export SYNCHRONAI_DIR="${SYNCHRONAI_DIR:-/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI}"
CACHE_DIR="${SWEEP_CACHE_DIR:-$SYNCHRONAI_DIR/data/fnirs_perpair_window_cache}"
SAVE_DIR="${SWEEP_SAVE_DIR:-$SYNCHRONAI_DIR/runs/fnirs_perpair_${NAME}_tile}"

echo "=== [$SCRIPT_VERSION] fNIRS cached sweep: $NAME ==="
echo "  base_width=$BASE_WIDTH depth=$DEPTH epochs=$EPOCHS batch=$BATCH"
echo "  cache:  $CACHE_DIR"
echo "  save:   $SAVE_DIR"

export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
# Force the ml-env interpreter (source activate is unreliable under LSF) + block ~/.local.
export PATH="$SYNCHRONAI_DIR/ml-env/bin:$PATH"
export PYTHONNOUSERSITE=1
# span[hosts=1] keeps the slots on one host; OMP/MKL make the math libs use them.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
cd "$SYNCHRONAI_DIR"
echo "Using python: $($SYNCHRONAI_DIR/ml-env/bin/python --version 2>&1)"

$SYNCHRONAI_DIR/ml-env/bin/python -m synchronai.main \
    --fnirs --train diffusion --per-pair \
    --data-dir     "$CACHE_DIR" \
    --window-cache "$CACHE_DIR" \
    --save-dir     "$SAVE_DIR" \
    --unet-base-width "$BASE_WIDTH" \
    --unet-depth "$DEPTH" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --learning-rate 1e-4 \
    --lr-schedule cosine_restarts
echo "=== [$SCRIPT_VERSION] $NAME exited code=$? ==="

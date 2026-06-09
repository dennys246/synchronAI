#!/bin/bash
#BSUB -J fnirs_perpair_probe
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB] span[hosts=1]'
# NOTE: #BSUB directives MUST stay directly under the shebang, before any
# command (incl. the SCRIPT_VERSION= assignment below). LSF stops parsing
# directives at the first non-comment line; putting a command above them makes
# LSF ignore every #BSUB line and use the whole script text as the job name.

SCRIPT_VERSION="generative_fnirs_perpair_probe_bsub-v3"

# =============================================================================
# fNIRS Per-Pair Generative — VALIDATION / CONVERGENCE PROBE (tile windowing)
#
# Writes to a FRESH save dir (default runs/fnirs_perpair_${NAME}_tile_probe) so
# it never clobbers the existing runs/fnirs_perpair_* encoders that back the
# current results. Two intended uses, both via env overrides:
#
#   1) Tiny dry-run — confirm FID/MMD now logs (per_pair fix) without the old
#      broadcast error, before committing real compute:
#        export PROBE_NAME=micro PROBE_BASE_WIDTH=8 PROBE_DEPTH=2 \
#               PROBE_EPOCHS=2 PROBE_MAX_RECORDINGS=20 PROBE_EVAL_GEN_EVERY=1
#      Success = log shows "Epoch 1/2 ... FID=.. MMD=.." with NO
#      "Generative metrics failed at epoch" warning.
#
#   2) Convergence probe — set the epoch budget for the full tile-mode sweep by
#      watching where the FID/MMD curve plateaus on the small model:
#        export PROBE_NAME=small PROBE_BASE_WIDTH=16 PROBE_DEPTH=3 \
#               PROBE_EPOCHS=40 PROBE_MAX_RECORDINGS=0 PROBE_EVAL_GEN_EVERY=2
#
# Submit:
#   export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
#   export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
#   export LSF_DOCKER_PRESERVE_ENVIRONMENT=true
#   export PROBE_NAME=... PROBE_BASE_WIDTH=... PROBE_DEPTH=... PROBE_EPOCHS=... \
#          PROBE_MAX_RECORDINGS=... PROBE_EVAL_GEN_EVERY=...
#   bsub -oo scripts/bsub/logs/fnirs_perpair_probe_$(date +'%m-%d').log \
#        < scripts/bsub/generative_fnirs_perpair_probe_bsub.sh
#
# NOTE: requires the windowing change (windowing.py + train.py FID fix) to be
# committed AND pushed first — the cluster imports synchronai via PYTHONPATH.
# =============================================================================

NAME="${PROBE_NAME:-small}"
BASE_WIDTH="${PROBE_BASE_WIDTH:-16}"
DEPTH="${PROBE_DEPTH:-3}"
EPOCHS="${PROBE_EPOCHS:-40}"
MAX_RECORDINGS="${PROBE_MAX_RECORDINGS:-0}"
EVAL_GEN_EVERY="${PROBE_EVAL_GEN_EVERY:-2}"
SAVE_DIR_DEFAULT="runs/fnirs_perpair_${NAME}_tile_probe"
SAVE_DIR="${PROBE_SAVE_DIR:-$SAVE_DIR_DEFAULT}"

echo "=== [$SCRIPT_VERSION] ==="
echo "=== fNIRS Per-Pair PROBE: $NAME (tile windowing) ==="
echo "  Base width:        $BASE_WIDTH"
echo "  Depth:             $DEPTH"
echo "  Epochs:            $EPOCHS"
echo "  Max recordings:    $MAX_RECORDINGS (0 = all)"
echo "  Eval gen every:    $EVAL_GEN_EVERY"
echo "  Save dir:          $SAVE_DIR"

export SYNCHRONAI_DIR="${SYNCHRONAI_DIR:-/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI}"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
conda init
source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR

# span[hosts=1] keeps all -n slots on one host; these exports make the math
# libraries actually use them. Without both, LSF scatters the slots and the CPU
# trainer only uses the master host's cores (slow / silent-hang risk).
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40

bash $SYNCHRONAI_DIR/scripts/generative_pretrain.sh \
    --save-dir "$SYNCHRONAI_DIR/$SAVE_DIR" \
    --unet-base-width "$BASE_WIDTH" \
    --unet-depth "$DEPTH" \
    --duration-seconds 60 \
    --epochs "$EPOCHS" \
    --max-recordings "$MAX_RECORDINGS" \
    --lr 1e-4 \
    --lr-schedule cosine_restarts \
    --eval-gen-every "$EVAL_GEN_EVERY" \
    --per-pair \
    --enable-qc \
    --sci-threshold 0.40 \
    --snr-threshold 2.0 \
    --cardiac-peak-ratio 2.0 \
    --no-require-cardiac

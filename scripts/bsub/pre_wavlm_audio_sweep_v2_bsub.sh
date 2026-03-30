#!/bin/sh
# =============================================================================
# Submit WavLM Audio Sweep v2 — Per-Layer Features + Smaller Architectures
#
# Sweep v1 showed all models peaked at epoch 1 with AUC ~0.678, suggesting
# the blended 768-dim features lack discriminative signal and the heads
# overfit immediately. This v2 sweep addresses both issues:
#
# 1. Per-layer features: learnable layer weights let the model discover
#    which WavLM layers carry synchrony signal (hypothesis: middle layers
#    encoding prosody/rhythm matter most)
#
# 2. Projection bottleneck: compress 768-dim → 128/256 before temporal
#    aggregation, forcing the model to find a compact representation
#    (mirrors DINOv2 finding that smaller architectures generalize better)
#
# 3. Smaller heads: hidden_dim=32/64 instead of 128, with higher dropout
#
# Step 1: Extract per-layer features (if not already done)
# Step 2: Submit 10 sweep training jobs
#
# After all jobs complete:
#   python scripts/compare_sweep_results.py --sweep-dir runs/audio_sweep_v2
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
EXTRACT_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/wavlm_extract_bsub.sh"
SWEEP_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/wavlm_audio_sweep_bsub.sh"
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

PERLAYER_FEATURES="data/wavlm_baseplus_perlayer_features"
BLENDED_FEATURES="data/wavlm_baseplus_features"
EXTRACT_JOB_NAME="synchronai-wavlm-extract-v2-$DATE"

echo "=========================================="
echo "  WavLM Audio Sweep v2"
echo "  Per-layer features + smaller architectures"
echo "  Date: $DATE"
echo "=========================================="

# =============================================================================
# NOTE: Extraction must be completed BEFORE running this script.
# The extraction job (wavlm_extract_bsub.sh) should have already finished
# and produced feature_index.csv in both feature directories.
# =============================================================================

echo ""
echo "Submitting 10 training jobs (no extraction dependency)..."

# Shared sweep config
export SWEEP_ENCODER="wavlm-base-plus"
export SWEEP_OUTPUT_BASE="runs/audio_sweep_v2"

# --- GROUP A: Per-layer features (learnable layer weights) ---

export SWEEP_FEATURE_DIR="$PERLAYER_FEATURES"

# --- Run 1: perlayer_proj128 — per-layer + projection to 128 ---
export SWEEP_RUN_NAME="perlayer_proj128"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="3e-4"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"
export SWEEP_PROJECT_DIM="128"

echo "  1/10: ${SWEEP_RUN_NAME} (perlayer, proj=128, h=64)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 2: perlayer_proj64 — even smaller projection ---
export SWEEP_RUN_NAME="perlayer_proj64"
export SWEEP_PROJECT_DIM="64"
export SWEEP_HIDDEN_DIM="32"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="3e-4"
export SWEEP_WEIGHT_DECAY="1e-2"

echo "  2/10: ${SWEEP_RUN_NAME} (perlayer, proj=64, h=32)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 3: perlayer_proj256 — moderate projection ---
export SWEEP_RUN_NAME="perlayer_proj256"
export SWEEP_PROJECT_DIM="256"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"

echo "  3/10: ${SWEEP_RUN_NAME} (perlayer, proj=256, h=64)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 4: perlayer_proj128_heavyreg — projection + heavy regularization ---
export SWEEP_RUN_NAME="perlayer_proj128_heavyreg"
export SWEEP_PROJECT_DIM="128"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_LABEL_SMOOTHING="0.05"

echo "  4/10: ${SWEEP_RUN_NAME} (perlayer, proj=128, h=64, d=0.7, ls=0.05)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 5: perlayer_proj128_mixup — projection + mixup ---
export SWEEP_RUN_NAME="perlayer_proj128_mixup"
export SWEEP_PROJECT_DIM="128"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="3e-4"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.3"

echo "  5/10: ${SWEEP_RUN_NAME} (perlayer, proj=128, h=64, mixup=0.3)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- GROUP B: Blended features + smaller architectures ---
# (uses existing blended features but with projection bottleneck)

export SWEEP_FEATURE_DIR="$BLENDED_FEATURES"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"

# --- Run 6: blended_proj128 — projection bottleneck on blended ---
export SWEEP_RUN_NAME="blended_proj128"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_PROJECT_DIM="128"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="3e-4"
export SWEEP_WEIGHT_DECAY="1e-2"

echo "  6/10: ${SWEEP_RUN_NAME} (blended, proj=128, h=64)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 7: blended_proj64 — very small ---
export SWEEP_RUN_NAME="blended_proj64"
export SWEEP_PROJECT_DIM="64"
export SWEEP_HIDDEN_DIM="32"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="3e-4"
export SWEEP_WEIGHT_DECAY="1e-2"

echo "  7/10: ${SWEEP_RUN_NAME} (blended, proj=64, h=32)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 8: blended_tiny — smallest possible ---
export SWEEP_RUN_NAME="blended_tiny"
export SWEEP_PROJECT_DIM="32"
export SWEEP_HIDDEN_DIM="16"
export SWEEP_DROPOUT="0.3"
export SWEEP_LEARNING_RATE="1e-3"
export SWEEP_WEIGHT_DECAY="1e-2"

echo "  8/10: ${SWEEP_RUN_NAME} (blended, proj=32, h=16)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 9: blended_proj128_heavyreg — strong regularization ---
export SWEEP_RUN_NAME="blended_proj128_heavyreg"
export SWEEP_PROJECT_DIM="128"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_LABEL_SMOOTHING="0.05"

echo "  9/10: ${SWEEP_RUN_NAME} (blended, proj=128, h=64, d=0.7, ls=0.05)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

# --- Run 10: blended_proj128_attention — attention + projection ---
export SWEEP_RUN_NAME="blended_proj128_attention"
export SWEEP_TEMPORAL_AGG="attention"
export SWEEP_PROJECT_DIM="128"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_LABEL_SMOOTHING="0.0"

echo "  10/10: ${SWEEP_RUN_NAME} (blended, attention, proj=128, h=64)"
bsub -J "synchronai-audio-v2-${SWEEP_RUN_NAME}-$DATE" \
     -oo "$LOG_DIR/audio_sweep_v2_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep_v2 \
     < "$SWEEP_SCRIPT"

echo ""
echo "=========================================="
echo "  10 training jobs submitted"
echo "  Assumes extraction already completed"
echo ""
echo "  Monitor with: bjobs -g /$USER/audio_sweep_v2"
echo ""
echo "  After all complete:"
echo "    python scripts/compare_sweep_results.py --sweep-dir runs/audio_sweep_v2"
echo "=========================================="

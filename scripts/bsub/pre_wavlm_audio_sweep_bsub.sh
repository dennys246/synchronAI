#!/bin/sh
# =============================================================================
# Submit WavLM Audio Feature Extraction + Sweep
#
# Step 1: Extract WavLM-base-plus features (GPU, ~99GB mem)
# Step 2: Submit 8 sweep training jobs (CPU, ~16GB mem each)
#         All training jobs depend on extraction completing first.
#
# WavLM-base-plus (768-dim, 12 layers) is used instead of Whisper because:
# - Native 1s chunk processing (no 30s padding waste)
# - Utterance mixing pretraining (ideal for dyadic conversation)
# - 768-dim features better matched to ~59K sample dataset
#
# After all jobs complete, compare results:
#   python scripts/compare_sweep_results.py --sweep-dir runs/audio_sweep
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

WAVLM_FEATURES="data/wavlm_baseplus_features"
EXTRACT_JOB_NAME="synchronai-wavlm-extract-$DATE"

echo "=========================================="
echo "  WavLM Audio Feature Sweep"
echo "  Encoder: WavLM-base-plus (768-dim)"
echo "  Date: $DATE"
echo "=========================================="

# =============================================================================
# Step 1: Submit feature extraction job
# =============================================================================

echo ""
echo "Submitting feature extraction job..."
bsub -J "$EXTRACT_JOB_NAME" \
     -oo "$LOG_DIR/wavlm_extract_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$EXTRACT_SCRIPT"

echo "  Extraction job: $EXTRACT_JOB_NAME"

# =============================================================================
# Step 2: Submit sweep training jobs (depend on extraction)
# =============================================================================

echo ""
echo "Submitting 8 training jobs (will wait for extraction)..."

# Shared sweep config
export SWEEP_ENCODER="wavlm-base-plus"
export SWEEP_FEATURE_DIR="$WAVLM_FEATURES"

# --- Run 1: baseline — mean pool, standard config ---
export SWEEP_RUN_NAME="baseline"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"

echo "  1/8: ${SWEEP_RUN_NAME} (mean h=128, d=0.5, lr=1e-4, wd=1e-3)"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 2: heavy_reg — max regularization ---
export SWEEP_RUN_NAME="heavy_reg"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"

echo "  2/8: ${SWEEP_RUN_NAME} (mean h=128, d=0.7, wd=1e-2)"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 3: lstm — LSTM temporal aggregation ---
export SWEEP_RUN_NAME="lstm"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"

echo "  3/8: ${SWEEP_RUN_NAME} (lstm h=128, d=0.5, lr=1e-4)"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 4: attention — attention temporal aggregation ---
export SWEEP_RUN_NAME="attention"
export SWEEP_TEMPORAL_AGG="attention"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"

echo "  4/8: ${SWEEP_RUN_NAME} (attention h=128, d=0.5, lr=1e-4)"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 5: small_cap — reduced capacity ---
export SWEEP_RUN_NAME="small_cap"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_HIDDEN_DIM="64"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="3e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"

echo "  5/8: ${SWEEP_RUN_NAME} (mean h=64, d=0.5, lr=3e-4)"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 6: mixup — feature mixup augmentation ---
export SWEEP_RUN_NAME="mixup"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.3"

echo "  6/8: ${SWEEP_RUN_NAME} (mean h=128, d=0.5, mixup=0.3)"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 7: label_smooth — label smoothing ---
export SWEEP_RUN_NAME="label_smooth"
export SWEEP_TEMPORAL_AGG="mean"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.5"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-3"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.05"
export SWEEP_MIXUP_ALPHA="0.0"

echo "  7/8: ${SWEEP_RUN_NAME} (mean h=128, d=0.5, label_smooth=0.05)"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

# --- Run 8: lstm_heavy_reg — best DINOv2 recipe adapted for audio ---
export SWEEP_RUN_NAME="lstm_heavy_reg"
export SWEEP_TEMPORAL_AGG="lstm"
export SWEEP_HIDDEN_DIM="128"
export SWEEP_DROPOUT="0.7"
export SWEEP_LEARNING_RATE="1e-4"
export SWEEP_WEIGHT_DECAY="1e-2"
export SWEEP_PATIENCE="15"
export SWEEP_LABEL_SMOOTHING="0.0"
export SWEEP_MIXUP_ALPHA="0.0"

echo "  8/8: ${SWEEP_RUN_NAME} (lstm h=128, d=0.7, wd=1e-2) — DINOv2 best recipe"
bsub -J "synchronai-audio-${SWEEP_RUN_NAME}-$DATE" \
     -w "done($EXTRACT_JOB_NAME)" \
     -oo "$LOG_DIR/audio_sweep_${SWEEP_RUN_NAME}_$DATE.log" \
     -g /$USER/audio_sweep \
     < "$SWEEP_SCRIPT"

echo ""
echo "=========================================="
echo "  1 extraction + 8 training jobs submitted"
echo "  Training jobs wait for extraction to finish"
echo ""
echo "  Monitor with: bjobs -g /$USER/audio_sweep"
echo ""
echo "  After all complete:"
echo "    python scripts/compare_sweep_results.py --sweep-dir runs/audio_sweep"
echo "=========================================="

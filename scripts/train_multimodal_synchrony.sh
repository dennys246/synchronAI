#!/bin/bash
# Train multi-modal classifier for synchrony detection
# Combines video (YOLO) and audio (Whisper) modalities
# Usage: ./scripts/train_multimodal_synchrony.sh

set -e  # Exit on error

# =============================================================================
# Directory Configuration
# =============================================================================

# Label directory: contains subject folders with session subfolders and labels
LABEL_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/synchrony_coding/archive/OLD_synchronycoding_participants/"

# Video directory: contains 4-digit subject prefix folders with session videos
VIDEO_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"

# Output directories
OUTPUT_DIR="runs/multimodal_classifier"
LABELS_CSV="data/labels.csv"

# =============================================================================
# Transfer Learning Configuration
# =============================================================================
#
# Three strategies available:
#
# 1. DEFAULT (pretrained backbones from Ultralytics/OpenAI):
#    PRETRAINED_VIDEO=""
#    PRETRAINED_AUDIO=""
#    → Uses pretrained YOLO + Whisper backbones, trains heads from scratch
#
# 2. LOAD HEADS ONLY (RECOMMENDED if you have trained models):
#    PRETRAINED_VIDEO="runs/video_classifier/best.pt"
#    PRETRAINED_AUDIO="runs/audio_classifier/best.pt"
#    LOAD_HEADS_ONLY="true"
#    → Keeps strong pretrained YOLO/Whisper backbones
#    → Uses your task-adapted heads from separate training
#
# 3. LOAD COMPLETE MODELS:
#    PRETRAINED_VIDEO="runs/video_classifier/best.pt"
#    PRETRAINED_AUDIO="runs/audio_classifier/best.pt"
#    LOAD_HEADS_ONLY="false"
#    → Loads everything from your trained models
#
PRETRAINED_VIDEO=""  # Path to your trained video model (optional)
PRETRAINED_AUDIO=""  # Path to your trained audio model (optional)
LOAD_HEADS_ONLY="true"  # true = keep pretrained backbones, false = load everything

# Video configuration
VIDEO_BACKBONE="yolo26s"
VIDEO_TEMPORAL_AGG="lstm"

# Audio configuration
# Whisper: large-v3, large-v2, large, medium, small, base, tiny
# WavLM (recommended): wavlm-large, wavlm-base-plus
AUDIO_MODEL="large-v3"

# Fusion configuration
FUSION_TYPE="cross_attention"  # concat, cross_attention, gated

# =============================================================================
# Training Configuration
# =============================================================================

SEED=42
BATCH_SIZE=16
EPOCHS=50
STAGE1_EPOCHS=5  # Fusion head only

# Learning rates
LEARNING_RATE=1e-4  # Stage 1
VIDEO_BACKBONE_LR=1e-5  # Stage 2
AUDIO_ENCODER_LR=1e-5  # Stage 2
FUSION_HEAD_LR=5e-5  # Stage 2

# Multi-task loss weights
SYNC_LOSS_WEIGHT=0.6  # Primary task
EVENT_LOSS_WEIGHT=0.4  # Auxiliary task

# Heatmap visualization (optional, helpful for debugging)
HEATMAP_EPOCH_INTERVAL=0  # Generate every N epochs (0 = disabled, e.g., 5 = every 5 epochs)
HEATMAP_VIDEO=""  # Path to sample video (auto-selected if empty)

# =============================================================================
# Step 1: Preprocess raw data to labels.csv (if not already done)
# =============================================================================

if [ ! -f "${LABELS_CSV}" ]; then
    echo "=== Step 1: Preprocessing raw data ==="
    echo "Label directory: ${LABEL_DIRECTORY}"
    echo "Video directory: ${VIDEO_DIRECTORY}"
    echo "Output CSV: ${LABELS_CSV}"

    python -m synchronai.main --preprocess \
        --label-dir "${LABEL_DIRECTORY}" \
        --video-dir "${VIDEO_DIRECTORY}" \
        --output-csv "${LABELS_CSV}" \
        --conflict-strategy last \
        --label-encoding "a:0,s:1"

    echo "Preprocessing complete. Labels saved to ${LABELS_CSV}"
else
    echo "=== Labels file already exists: ${LABELS_CSV} ==="
    echo "Skipping preprocessing step."
fi

# =============================================================================
# Step 2: Train multi-modal classifier
# =============================================================================

echo ""
echo "=== Training Multi-Modal Synchrony Classifier ==="
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Model Configuration:"
echo "  Video backbone: ${VIDEO_BACKBONE}"
echo "  Audio model: ${AUDIO_MODEL}"
echo "  Fusion type: ${FUSION_TYPE}"
echo ""
echo "Training Configuration:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Total epochs: ${EPOCHS}"
echo "  Stage 1 epochs: ${STAGE1_EPOCHS} (fusion head only)"
echo "  Stage 2 epochs: $((EPOCHS - STAGE1_EPOCHS)) (full fine-tuning)"
echo ""
echo "Learning Rates:"
echo "  Stage 1: ${LEARNING_RATE}"
echo "  Stage 2 - Video backbone: ${VIDEO_BACKBONE_LR}"
echo "  Stage 2 - Audio encoder: ${AUDIO_ENCODER_LR}"
echo "  Stage 2 - Fusion/heads: ${FUSION_HEAD_LR}"
echo ""
echo "Loss Weights:"
echo "  Synchrony: ${SYNC_LOSS_WEIGHT}"
echo "  Audio events: ${EVENT_LOSS_WEIGHT}"
echo ""

# Build training command
TRAIN_CMD="python -m synchronai.training.multimodal.train \
    --labels-file ${LABELS_CSV} \
    --save-dir ${OUTPUT_DIR} \
    --video-backbone ${VIDEO_BACKBONE} \
    --video-temporal-agg ${VIDEO_TEMPORAL_AGG} \
    --audio-model ${AUDIO_MODEL} \
    --fusion-type ${FUSION_TYPE} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --stage1-epochs ${STAGE1_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --video-backbone-lr ${VIDEO_BACKBONE_LR} \
    --audio-encoder-lr ${AUDIO_ENCODER_LR} \
    --fusion-head-lr ${FUSION_HEAD_LR} \
    --sync-loss-weight ${SYNC_LOSS_WEIGHT} \
    --event-loss-weight ${EVENT_LOSS_WEIGHT} \
    --seed ${SEED} \
    --num-workers 4 \
    --use-amp"

# Add heatmap options if enabled
if [ "${HEATMAP_EPOCH_INTERVAL}" -gt 0 ]; then
    TRAIN_CMD="${TRAIN_CMD} --heatmap-epoch-interval ${HEATMAP_EPOCH_INTERVAL}"
    echo "Heatmap Generation:"
    echo "  Interval: Every ${HEATMAP_EPOCH_INTERVAL} epochs"
    if [ -n "${HEATMAP_VIDEO}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --heatmap-video ${HEATMAP_VIDEO}"
        echo "  Video: ${HEATMAP_VIDEO}"
    else
        echo "  Video: (auto-select from training data)"
    fi
    echo ""
fi

# Add pretrained model paths if specified
if [ -n "${PRETRAINED_VIDEO}" ] || [ -n "${PRETRAINED_AUDIO}" ]; then
    echo "Transfer Learning Strategy:"
    if [ "${LOAD_HEADS_ONLY}" = "true" ]; then
        echo "  → Load heads only (keep pretrained YOLO/Whisper backbones)"
        TRAIN_CMD="${TRAIN_CMD} --load-heads-only"
    else
        echo "  → Load complete models (backbones + heads)"
    fi
fi

if [ -n "${PRETRAINED_VIDEO}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pretrained-video ${PRETRAINED_VIDEO}"
    echo "  Video checkpoint: ${PRETRAINED_VIDEO}"
fi

if [ -n "${PRETRAINED_AUDIO}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pretrained-audio ${PRETRAINED_AUDIO}"
    echo "  Audio checkpoint: ${PRETRAINED_AUDIO}"
fi

# Run training
echo ""
echo "Starting training..."
eval "${TRAIN_CMD}"

echo ""
echo "=== Training complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "Latest checkpoint: ${OUTPUT_DIR}/latest.pt"
echo "Training history: ${OUTPUT_DIR}/history.json"

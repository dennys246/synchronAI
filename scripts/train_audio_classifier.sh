#!/bin/bash
# Audio classifier training script
# Trains the Whisper-based audio event classifier

set -e

# Get project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration - use absolute paths based on project root
# NOTE: This script expects real audio event labels (not synchrony labels).
LABELS_FILE="${PROJECT_ROOT}/data/audio_event_labels.csv"
SAVE_DIR="${PROJECT_ROOT}/runs/audio_classifier"
WHISPER_MODEL="large-v3"

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE=1e-4
VAL_SPLIT=0.2

echo "=========================================="
echo "Audio Classifier Training"
echo "=========================================="
echo "Labels file: $LABELS_FILE"
echo "Save directory: $SAVE_DIR"
echo "Whisper model: $WHISPER_MODEL"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="

# Create save directory
mkdir -p "$SAVE_DIR"

# Run training
python -m synchronai.main \
    --train audio-classifier \
    --audio-labels-file "$LABELS_FILE" \
    --save-dir "$SAVE_DIR" \
    --whisper-model "$WHISPER_MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --val-split "$VAL_SPLIT" \
    --use-amp

echo "=========================================="
echo "Training complete!"
echo "Model saved to: $SAVE_DIR/best.pt"
echo "Training plot: $SAVE_DIR/training_plot.png"
echo "=========================================="

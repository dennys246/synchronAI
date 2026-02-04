#!/bin/bash
# Train video classifier for synchrony detection
# Usage: ./scripts/train_cv_synchrony.sh

set -e  # Exit on error

# =============================================================================
# Directory Configuration
# =============================================================================

# Label directory: contains subject folders (e.g., 50001/, 50021/) with session
# subfolders (V0/, V1/, V2/) containing xlsx label files
LABEL_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/synchrony_coding/archive/OLD_synchronycoding_participants/"

# Video directory: contains 4-digit subject prefix folders (e.g., 5000/, 5002/)
# with session subfolders containing mp4 files
VIDEO_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"

# Output directories (standardized path for easier resume)
OUTPUT_DIR="runs/video_classifier"
LABELS_CSV="data/labels.csv"

# =============================================================================
# Training Configuration
# =============================================================================

SEED=42
BATCH_SIZE=16
EPOCHS=50
BACKBONE="yolo26n"
BACKBONE_TASK="pose"  # Use pose weights for human-centric task
TEMPORAL_AGG="attention"

# Resume training from checkpoint (set to "latest.pt" to resume)
RESUME_FROM=""  # e.g., "latest.pt"

# Heatmap visualization during training
# Set HEATMAP_INTERVAL to generate heatmaps every N batches (0 = disabled)
HEATMAP_INTERVAL=10
# Path to a sample video for heatmap generation (leave empty to auto-select from training data)
HEATMAP_VIDEO=""  # Leave empty to auto-select, or set to specific video path

# =============================================================================
# Step 1: Preprocess raw data to labels.csv
# =============================================================================

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

# =============================================================================
# Step 2: Validate dataset
# =============================================================================

echo ""
echo "=== Step 2: Validating dataset ==="

python -m synchronai.main --video --validate \
    --labels-file "${LABELS_CSV}" \
    --video-dir "${VIDEO_DIRECTORY}"

# =============================================================================
# Step 3: Train video classifier
# =============================================================================

echo ""
echo "=== Step 3: Training video classifier ==="
echo "Output directory: ${OUTPUT_DIR}"

# Build training command
TRAIN_CMD="python -m synchronai.main --video --train classifier \
    --labels-file ${LABELS_CSV} \
    --save-dir ${OUTPUT_DIR} \
    --backbone ${BACKBONE} \
    --backbone-task ${BACKBONE_TASK} \
    --temporal-aggregation ${TEMPORAL_AGG} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --num-workers 0 \
    --use-amp"

# Add heatmap options if configured
if [ "${HEATMAP_INTERVAL}" -gt 0 ]; then
    TRAIN_CMD="${TRAIN_CMD} --heatmap-batch-interval ${HEATMAP_INTERVAL}"
    echo "  Heatmap interval: every ${HEATMAP_INTERVAL} batches"
    if [ -n "${HEATMAP_VIDEO}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --heatmap-video ${HEATMAP_VIDEO}"
        echo "  Heatmap video: ${HEATMAP_VIDEO}"
    else
        echo "  Heatmap video: (auto-select from training data)"
    fi
fi

# Add resume option if specified
if [ -n "${RESUME_FROM}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume-from ${RESUME_FROM}"
    echo "  Resuming from: ${RESUME_FROM}"
fi

# Run training
eval "${TRAIN_CMD}"

echo ""
echo "=== Training complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "Latest checkpoint: ${OUTPUT_DIR}/latest.pt"
echo "Training plot: ${OUTPUT_DIR}/training_plot.png"
echo "Training history: ${OUTPUT_DIR}/history.json"

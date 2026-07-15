#!/bin/bash
# Two-step DINOv2 training pipeline:
# 1. Extract DINOv2 features from videos (one-time, CPU-friendly)
# 2. Train LSTM + head on pre-extracted features (fast, CPU-friendly)
#
# This approach is mathematically identical to live DINOv2 training for
# stage 1 (frozen backbone), but runs ~100x faster since DINOv2 inference
# is done once and cached.

set -e  # Exit on error

# =============================================================================
# Directory Configuration
# =============================================================================

LABEL_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/synchrony_coding/archive/OLD_synchronycoding_participants/"
VIDEO_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"

LABELS_CSV="data/labels.csv"
FEATURE_DIR="data/dinov2_features_meanpatch"
OUTPUT_DIR="runs/dinov2_features"

# =============================================================================
# Feature Extraction Configuration
# =============================================================================

BACKBONE="dinov2-base"        # DINOv2 variant: dinov2-small, dinov2-base, dinov2-large
FRAME_SIZE=224                # DINOv2 native input size
SAMPLE_FPS=12                 # Frames per second to sample
WINDOW_SECONDS=1.0            # Window duration (matches live training)
POOL_MODE="mean_patch"        # Feature pooling: cls (CLS token) or mean_patch (mean of patches)

# =============================================================================
# Training Configuration
# =============================================================================

SEED=42
TEMPORAL_AGG="lstm"           # mean, max, attention, lstm
HIDDEN_DIM=128
DROPOUT=0.5
BATCH_SIZE=64                 # Can be large since features are small tensors
EPOCHS=50
LEARNING_RATE=3e-5
WEIGHT_DECAY=1e-3
PATIENCE=15                   # Early stopping patience

# =============================================================================
# Step 1: Preprocess raw data to labels.csv (skip if already exists)
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
# Step 2: Extract DINOv2 features (skip if already done)
# =============================================================================

if [ ! -f "${FEATURE_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Step 2: Extracting DINOv2 features ==="
    echo "Backbone: ${BACKBONE}"
    echo "Frame size: ${FRAME_SIZE}x${FRAME_SIZE}"
    echo "Sample FPS: ${SAMPLE_FPS}, Window: ${WINDOW_SECONDS}s"
    echo "Output directory: ${FEATURE_DIR}"

    python scripts/extract_dinov2_features.py \
        --labels-file "${LABELS_CSV}" \
        --output-dir "${FEATURE_DIR}" \
        --backbone "${BACKBONE}" \
        --sample-fps ${SAMPLE_FPS} \
        --window-seconds ${WINDOW_SECONDS} \
        --frame-size ${FRAME_SIZE} \
        --pool-mode ${POOL_MODE} \
        --device auto

    echo "Feature extraction complete."
else
    echo ""
    echo "=== Features already extracted: ${FEATURE_DIR}/feature_index.csv ==="
    echo "Skipping extraction step."
fi

# =============================================================================
# Step 3: Train LSTM + head on pre-extracted features
# =============================================================================

echo ""
echo "=== Step 3: Training LSTM + head on pre-extracted features ==="
echo "Feature directory: ${FEATURE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Temporal aggregation: ${TEMPORAL_AGG}"
echo "Hidden dim: ${HIDDEN_DIM}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"

python scripts/train_from_features.py \
    --feature-dir "${FEATURE_DIR}" \
    --save-dir "${OUTPUT_DIR}" \
    --backbone "${BACKBONE}" \
    --temporal-aggregation ${TEMPORAL_AGG} \
    --hidden-dim ${HIDDEN_DIM} \
    --dropout ${DROPOUT} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --patience ${PATIENCE} \
    --seed ${SEED} \
    --num-workers 0

echo ""
echo "=== Training complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "Latest checkpoint: ${OUTPUT_DIR}/latest.pt"
echo "Training plot: ${OUTPUT_DIR}/training_plot.png"
echo "Training history: ${OUTPUT_DIR}/history.json"

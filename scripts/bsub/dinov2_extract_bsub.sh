#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# =============================================================================
# DINOv2 Feature Extraction — Prerequisite for Sweep Training
#
# Extracts mean_patch features for both dinov2-base and dinov2-small.
# Also builds the ml-env virtual environment so sweep training jobs can
# use it without conflicting pip installs.
#
# Must complete BEFORE sweep training jobs start (enforced via bsub -w).
# =============================================================================

cd $SYNCHRONAI_DIR

# Redirect HuggingFace cache to storage (home directory has limited quota)
export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"

# =============================================================================
# Step 1: Build ml-env from scratch
# =============================================================================

echo "=== Building ml-env virtual environment ==="

# Remove stale env to avoid conflicts
if [ -d "$SYNCHRONAI_DIR/ml-env" ]; then
    echo "Removing existing ml-env..."
    rm -rf "$SYNCHRONAI_DIR/ml-env"
fi

python -m venv "$SYNCHRONAI_DIR/ml-env"
source "$SYNCHRONAI_DIR/ml-env/bin/activate"

echo "Installing core dependencies..."
pip install --upgrade pip
pip install --no-cache-dir "numpy>=2.0,<2.5"
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir transformers scikit-learn tqdm pandas matplotlib

# Headless OpenCV for video reading in Docker (no libGL)
pip install --no-cache-dir opencv-python-headless

# Install synchronAI package in editable mode
pip install -e .

echo "ml-env built successfully."
python -c "import torch; import cv2; import sklearn; print('All imports OK')"

# =============================================================================
# Step 2: Preprocess labels (skip if already exists)
# =============================================================================

LABELS_CSV="data/labels.csv"
LABEL_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/synchrony_coding/archive/OLD_synchronycoding_participants/"
VIDEO_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"

if [ ! -f "${LABELS_CSV}" ]; then
    echo ""
    echo "=== Preprocessing raw data ==="

    python -m synchronai.main --preprocess \
        --label-dir "${LABEL_DIRECTORY}" \
        --video-dir "${VIDEO_DIRECTORY}" \
        --output-csv "${LABELS_CSV}" \
        --conflict-strategy last \
        --label-encoding "a:0,s:1"
else
    echo ""
    echo "=== Labels file already exists: ${LABELS_CSV} ==="
fi

# =============================================================================
# Step 3: Extract dinov2-base mean_patch features
# =============================================================================

BASE_FEATURE_DIR="data/dinov2_features_meanpatch"

if [ ! -f "${BASE_FEATURE_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting dinov2-base mean_patch features ==="

    python scripts/extract_dinov2_features.py \
        --labels-file "${LABELS_CSV}" \
        --output-dir "${BASE_FEATURE_DIR}" \
        --backbone dinov2-base \
        --sample-fps 12 \
        --window-seconds 1.0 \
        --frame-size 224 \
        --pool-mode mean_patch \
        --device auto

    echo "dinov2-base extraction complete."
else
    echo ""
    echo "=== dinov2-base features already extracted ==="
fi

# =============================================================================
# Step 4: Extract dinov2-small mean_patch features (for small_backbone run)
# =============================================================================

SMALL_FEATURE_DIR="data/dinov2_features_small_meanpatch"

if [ ! -f "${SMALL_FEATURE_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting dinov2-small mean_patch features ==="

    python scripts/extract_dinov2_features.py \
        --labels-file "${LABELS_CSV}" \
        --output-dir "${SMALL_FEATURE_DIR}" \
        --backbone dinov2-small \
        --sample-fps 12 \
        --window-seconds 1.0 \
        --frame-size 224 \
        --pool-mode mean_patch \
        --device auto

    echo "dinov2-small extraction complete."
else
    echo ""
    echo "=== dinov2-small features already extracted ==="
fi

echo ""
echo "=== Feature extraction complete ==="
echo "dinov2-base features:  ${BASE_FEATURE_DIR}"
echo "dinov2-small features: ${SMALL_FEATURE_DIR}"
echo "ml-env ready for sweep training jobs."
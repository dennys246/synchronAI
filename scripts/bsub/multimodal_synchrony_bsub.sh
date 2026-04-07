#!/bin/bash
SCRIPT_VERSION="multimodal_synchrony_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate

cd $SYNCHRONAI_DIR

# =============================================================================
# Install Dependencies for Multi-Modal Training
# =============================================================================

echo "=== [$SCRIPT_VERSION] ==="
echo "=== Installing multi-modal dependencies ==="

# Fix NumPy installation - reinstall to get proper binary dependencies
# NumPy 2.x bundles OpenBLAS, but the wheel wasn't installed correctly
echo "Installing NumPy..."
pip install --force-reinstall --no-cache-dir "numpy>=2.0,<2.5"

# Clear any corrupted whisper wheel cache
echo "Clearing Whisper cache..."
rm -rf /home/$USER/.cache/pip/wheels/*/openai_whisper* 2>/dev/null || true

# Install audio dependencies (--no-cache-dir ensures fresh download)
echo "Installing audio dependencies..."
pip install --no-cache-dir openai-whisper soundfile imageio-ffmpeg
pip install --no-cache-dir transformers  # WavLM encoder support

# Install synchronAI package
echo "Installing synchronAI package..."
pip install -e .

# Fix OpenCV for headless Docker container (no libGL)
# MUST run AFTER pip install -e . because ultralytics pulls in opencv-python (non-headless)
# Uninstall BOTH variants first — they share the cv2 module directory, so partial
# uninstall leaves cv2 broken. Then fresh-install only the headless version.
echo "Replacing OpenCV with headless version..."
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install --force-reinstall opencv-python-headless

# =============================================================================
# Directory Configuration
# =============================================================================

LABEL_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/synchrony_coding/archive/OLD_synchronycoding_participants/"
VIDEO_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"
OUTPUT_DIR="runs/multimodal_classifier"
LABELS_CSV="data/labels.csv"
CONFIG_FILE="configs/train/multimodal_classifier.yaml"

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
# Step 2: Train multi-modal classifier using YAML config
# =============================================================================

echo ""
echo "=== Starting Multi-Modal Synchrony Classifier Training ==="
echo "Config file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

python -m synchronai.training.multimodal.train \
    --config "${CONFIG_FILE}" \
    --labels-file "${LABELS_CSV}" \
    --save-dir "${OUTPUT_DIR}"

echo ""
echo "=== Multi-Modal Training Complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "Latest checkpoint: ${OUTPUT_DIR}/latest.pt"
echo "Training history: ${OUTPUT_DIR}/history.json"

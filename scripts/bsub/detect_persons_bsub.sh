#!/bin/bash
SCRIPT_VERSION="detect_persons_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -J synchronai-detect-persons
#BSUB -o scripts/bsub/logs/detect_persons_%J.stdout
#BSUB -e scripts/bsub/logs/detect_persons_%J.stderr
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# Save LSF working directory before .bashrc may change it
PROJECT_DIR="$(pwd)"

source /home/$USER/.bashrc
source "${SYNCHRONAI_DIR:-$PROJECT_DIR}"/ml-env/bin/activate

cd "$PROJECT_DIR"

echo "=== [$SCRIPT_VERSION] ==="
echo "Working directory: $(pwd)"

# =============================================================================
# Install Dependencies for Person Detection
# =============================================================================

echo "=== Installing person detection dependencies ==="

# Fix NumPy installation
pip install --force-reinstall --no-cache-dir "numpy>=2.0,<2.5"

# Install synchronAI package
pip install -e .

# Install MMDet + dependencies for RTMDet person detection
pip install --no-cache-dir mmengine mmdet

# Fix OpenCV for headless Docker container
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install --force-reinstall opencv-python-headless

# =============================================================================
# Configuration
# =============================================================================

LABELS_CSV="${LABELS_CSV:-data/labels.csv}"
BBOX_OUTPUT_DIR="${BBOX_OUTPUT_DIR:-data/person_bboxes}"
DETECT_MODEL="${DETECT_MODEL:-rtmdet-m}"
CONFIDENCE="${CONFIDENCE:-0.5}"

# =============================================================================
# Run Person Detection
# =============================================================================

echo ""
echo "=== Running Offline Person Detection ==="
echo "Labels file: ${LABELS_CSV}"
echo "Output directory: ${BBOX_OUTPUT_DIR}"
echo "Model: ${DETECT_MODEL}"
echo "Confidence threshold: ${CONFIDENCE}"
echo ""

python scripts/detect_persons.py \
    --labels-file "${LABELS_CSV}" \
    --output-dir "${BBOX_OUTPUT_DIR}" \
    --model "${DETECT_MODEL}" \
    --confidence-threshold "${CONFIDENCE}" \
    --device cuda \
    --skip-existing

echo ""
echo "=== Person Detection Complete ==="
echo "Bounding box JSONs saved to: ${BBOX_OUTPUT_DIR}"

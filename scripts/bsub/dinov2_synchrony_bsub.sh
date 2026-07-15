#!/bin/bash
SCRIPT_VERSION="dinov2_synchrony_bsub-v1"
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

# Redirect HuggingFace cache to storage (home directory has limited quota)
export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"

# =============================================================================
# Install Dependencies for DINOv2 Training
# =============================================================================

echo "=== [$SCRIPT_VERSION] ==="
echo "=== Installing DINOv2 dependencies ==="

# Fix NumPy installation - reinstall to get proper binary dependencies
echo "Installing NumPy..."
pip install --force-reinstall --no-cache-dir "numpy>=2.0,<2.5"

# Install HuggingFace transformers (DINOv2 backbone)
echo "Installing transformers for DINOv2..."
pip install --no-cache-dir transformers

# Install synchronAI package
echo "Installing synchronAI package..."
pip install -e .

# Fix OpenCV for headless Docker container (no libGL)
# MUST run AFTER pip install -e . because ultralytics pulls in opencv-python (non-headless)
echo "Replacing OpenCV with headless version..."
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install --force-reinstall opencv-python-headless

# =============================================================================
# Run DINOv2 Training
# =============================================================================

bash $SYNCHRONAI_DIR/scripts/train_dinov2_synchrony.sh
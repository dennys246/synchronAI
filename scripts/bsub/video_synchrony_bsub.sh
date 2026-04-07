#!/bin/bash
SCRIPT_VERSION="video_synchrony_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

echo "=== [$SCRIPT_VERSION] ==="

source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate

cd $SYNCHRONAI_DIR

# Fix NumPy installation - reinstall to get proper binary dependencies
# NumPy 2.x bundles OpenBLAS, but the wheel wasn't installed correctly
pip install --force-reinstall --no-cache-dir "numpy>=2.0,<2.5"

# Fix OpenCV for headless Docker container (no libGL)
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install opencv-python-headless

bash $SYNCHRONAI_DIR/scripts/train_cv_synchrony.sh

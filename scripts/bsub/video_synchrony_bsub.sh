#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

source /home/dennys/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate

cd $SYNCHRONAI_DIR

# Fix OpenCV for headless Docker container (no libGL)
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install opencv-python-headless

bash $SYNCHRONAI_DIR/scripts/train_cv_synchrony.sh

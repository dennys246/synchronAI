#!/bin/bash
SCRIPT_VERSION="generative_fnirs_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

echo "=== [$SCRIPT_VERSION] ==="

conda init
source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR
pip install -e .

bash $SYNCHRONAI_DIR/scripts/generative_pretrain.sh --duration-seconds 60 \
  --enable-qc --sci-threshold 0.75 --snr-threshold 5.0 --cardiac-peak-ratio 2.0

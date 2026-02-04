#!/bin/bash 
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

conda init
source /home/dennys/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR
pip install -e .

bash $SYNCHRONAI_DIR/scripts/generative_pretrain.sh --duration-seconds 15

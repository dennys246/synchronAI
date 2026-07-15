#!/bin/bash
SCRIPT_VERSION="audio_synchrony_bsub-v1"
echo "=== [$SCRIPT_VERSION] ==="
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 20
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate

cd $SYNCHRONAI_DIR

# Clear any corrupted whisper wheel cache
rm -rf /home/$USER/.cache/pip/wheels/*/openai_whisper* 2>/dev/null || true

# Install audio dependencies (--no-cache-dir ensures fresh download)
pip install --no-cache-dir openai-whisper soundfile imageio-ffmpeg
pip install -e .

# Run audio classifier training (100 epochs)
bash $SYNCHRONAI_DIR/scripts/train_audio_classifier.sh

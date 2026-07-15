#!/bin/bash
SCRIPT_VERSION="multimodal_synchrony_bsub-v5"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# ml-env is maintained by the extraction scripts (wavlm_extract_bsub.sh,
# dinov2_extract_bsub.sh). This training script uses the absolute python path
# directly — `source activate` is unreliable inside LSF Docker heredocs and
# silently falls back to conda's python (causing ModuleNotFoundError: torch).
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

# HuggingFace cache on shared NFS, not /home/$USER (which has a quota).
export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"

# Make synchronai package importable without `pip install -e .` (which races
# across concurrent Docker containers on the shared ml-env). Set this here in
# the body, not just the launcher — Docker env propagation has been unreliable.
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

cd $SYNCHRONAI_DIR

echo "=== [$SCRIPT_VERSION] ==="

# =============================================================================
# Preflight: verify ml-env has required packages
# =============================================================================

echo "=== Preflight: checking ml-env imports ==="
"$ML_PY" -c "import torch, transformers, soundfile, cv2, synchronai; print(f'torch={torch.__version__} transformers={transformers.__version__} soundfile ok cv2 ok synchronai ok')" || {
    echo "ERROR: ml-env missing required packages or synchronai not on PYTHONPATH."
    exit 1
}

# =============================================================================
# Directory Configuration
# =============================================================================

LABEL_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/synchrony_coding/archive/OLD_synchronycoding_participants/"
VIDEO_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"
OUTPUT_DIR="runs/multimodal_dinov2_wavlm"
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

    "$ML_PY" -m synchronai.main --preprocess \
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

"$ML_PY" -m synchronai.training.multimodal.train \
    --config "${CONFIG_FILE}" \
    --labels-file "${LABELS_CSV}" \
    --save-dir "${OUTPUT_DIR}"

echo ""
echo "=== Multi-Modal Training Complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "Latest checkpoint: ${OUTPUT_DIR}/latest.pt"
echo "Training history: ${OUTPUT_DIR}/history.json"

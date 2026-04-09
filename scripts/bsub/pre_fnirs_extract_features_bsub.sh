#!/bin/bash
SCRIPT_VERSION="pre_fnirs_extract_features_bsub-v1"
# =============================================================================
# fNIRS Per-Pair Feature Extraction — All Model Sizes
#
# Extraction-only script. Converts TF weights to PyTorch (if needed) and
# extracts per-pair features for all 4 model sizes.
#
# Run BEFORE the training sweep (pre_fnirs_child_adult_sweep_bsub.sh).
# =============================================================================

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

echo "=========================================="
echo "  [$SCRIPT_VERSION]"
echo "  fNIRS Per-Pair Feature Extraction"
echo "  Date: $DATE"
echo "=========================================="

submit_extraction() {
    local MODEL_NAME="$1"
    local PRETRAIN_DIR="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}"
    local CONFIG_JSON="${PRETRAIN_DIR}/fnirs_diffusion_config.json"
    local WEIGHTS_H5="${PRETRAIN_DIR}/fnirs_unet.weights.h5"
    local ENCODER_PT="${PRETRAIN_DIR}/fnirs_unet_encoder.pt"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_features"

    echo ""
    echo "=== $MODEL_NAME ==="

    bsub -J "synchronai-extract-${MODEL_NAME}-$DATE" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 16000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 8 \
         -R 'select[mem>16GB] rusage[mem=16GB]' \
         -oo "$LOG_DIR/fnirs_extract_${MODEL_NAME}_$DATE.log" \
         -g /$USER/fnirs_extract \
         << EXTRACT_EOF
echo "=== [$SCRIPT_VERSION] extract $MODEL_NAME ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"

# Convert TF -> PyTorch if needed
if [ ! -f "$ENCODER_PT" ]; then
    if [ ! -f "$WEIGHTS_H5" ]; then
        echo "ERROR: Pretrained weights not found: $WEIGHTS_H5"
        exit 1
    fi
    echo "=== Converting TF weights to PyTorch ==="
    python scripts/convert_fnirs_tf_to_pt.py \
        --config-json "$CONFIG_JSON" \
        --weights-path "$WEIGHTS_H5" \
        --output "$ENCODER_PT" \
        --verify
    if [ \$? -ne 0 ]; then
        echo "ERROR: Weight conversion failed!"
        exit 1
    fi
fi

# Clear any empty/stale feature directory
rm -rf "$FEATURE_DIR"

echo "=== Extracting per-pair features ==="
python scripts/extract_fnirs_features.py \
    --encoder-weights "$ENCODER_PT" \
    --data-dirs "$FNIRS_DIRS" \
    --output-dir "$FEATURE_DIR" \
    --per-pair \
    --stride-seconds 60.0 \
    --enable-qc \
    --sci-threshold 0.40 \
    --snr-threshold 2.0 \
    --cardiac-peak-ratio 2.0 \
    --no-require-cardiac \
    --include-tiers "gold,standard,salvageable"

echo "=== Extraction complete for $MODEL_NAME ==="
echo "Feature count:"
if [ -f "$FEATURE_DIR/feature_index.csv" ]; then
    wc -l "$FEATURE_DIR/feature_index.csv"
    echo "Tier distribution:"
    tail -n +2 "$FEATURE_DIR/feature_index.csv" | awk -F',' '{print \$NF}' | sort | uniq -c
else
    echo "ERROR: No feature_index.csv produced!"
    exit 1
fi
EXTRACT_EOF
}

submit_extraction "micro"
submit_extraction "small"
submit_extraction "medium"
submit_extraction "large"

echo ""
echo "=========================================="
echo "  4 extraction jobs submitted"
echo "  Monitor: bjobs -g /\$USER/fnirs_extract"
echo ""
echo "  After all complete, run training:"
echo "    sh scripts/bsub/pre_fnirs_child_adult_sweep_bsub.sh"
echo "=========================================="

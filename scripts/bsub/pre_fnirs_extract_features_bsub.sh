#!/bin/bash
SCRIPT_VERSION="pre_fnirs_extract_features_bsub-v5"
# =============================================================================
# fNIRS Per-Pair Feature Extraction — All Model Sizes
#
# Uses pre-computed QC cache (from pre_fnirs_compute_qc_bsub.sh) for tier
# filtering, and batched encoder for speed. No inline QC.
#
# Run AFTER: pre_fnirs_compute_qc_bsub.sh (produces qc_tiers.csv)
# Run BEFORE: pre_fnirs_child_adult_sweep_bsub.sh (training)
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

QC_CACHE="$SYNCHRONAI_DIR/data/qc_tiers.csv"

echo "=========================================="
echo "  [$SCRIPT_VERSION]"
echo "  fNIRS Per-Pair Feature Extraction"
echo "  Date: $DATE"
echo "=========================================="

submit_extraction() {
    local MODEL_NAME="$1"
    local EXTRA_FLAGS="${2:-}"
    local PRETRAIN_DIR="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}"
    local CONFIG_JSON="${PRETRAIN_DIR}/fnirs_diffusion_config.json"
    local WEIGHTS_H5="${PRETRAIN_DIR}/fnirs_unet.weights.h5"
    local ENCODER_PT="${PRETRAIN_DIR}/fnirs_unet_encoder.pt"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_features"

    echo ""
    echo "=== $MODEL_NAME ${EXTRA_FLAGS} ==="

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
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"
# ml-env/bin/activate does not reliably work inside LSF heredocs on this
# cluster — invoke ml-env python by absolute path. See docs/ris_bsub_reference.md.
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

# Convert TF -> PyTorch if needed
if [ ! -f "$ENCODER_PT" ]; then
    if [ ! -f "$WEIGHTS_H5" ]; then
        echo "ERROR: Pretrained weights not found: $WEIGHTS_H5"
        exit 1
    fi
    echo "=== Converting TF weights to PyTorch ==="
    "\$ML_PY" scripts/convert_fnirs_tf_to_pt.py \
        --config-json "$CONFIG_JSON" \
        --weights-path "$WEIGHTS_H5" \
        --output "$ENCODER_PT" \
        --verify
    if [ \$? -ne 0 ]; then
        echo "ERROR: Weight conversion failed!"
        exit 1
    fi
fi

# Don't wipe the directory — resume support uses existing .pt files
# (Comment out the rm -rf so partial work from previous runs is preserved)
# rm -rf "$FEATURE_DIR"

echo "=== Extracting per-pair features ==="
"\$ML_PY" scripts/extract_fnirs_features.py \
    --encoder-weights "$ENCODER_PT" \
    --data-dirs "$FNIRS_DIRS" \
    --output-dir "$FEATURE_DIR" \
    --per-pair \
    --stride-seconds 60.0 \
    --qc-cache "$QC_CACHE" \
    --include-tiers "gold,standard,salvageable" \
    --encoder-batch-size 32 \
    --pack-output \
    --delete-unpacked \
    $EXTRA_FLAGS

echo "=== Extraction complete for $MODEL_NAME ==="
if [ -f "$FEATURE_DIR/feature_index.csv" ]; then
    echo "Feature count:"
    wc -l "$FEATURE_DIR/feature_index.csv"
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

#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# =============================================================================
# DINOv2 Multi-Resolution Feature Extraction
#
# Extracts dinov2-small features at 112×112 and 168×168 resolutions.
# (224×224 features already exist from the previous sweep extraction.)
#
# DINOv2 handles non-224 resolutions natively via position embedding
# interpolation. Feature dim (384 for small) stays the same across all
# resolutions — only the spatial information captured changes.
#
# Must complete BEFORE progressive training jobs start.
# =============================================================================

cd $SYNCHRONAI_DIR

export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"

source "$SYNCHRONAI_DIR/ml-env/bin/activate"

LABELS_CSV="data/labels.csv"

# =============================================================================
# Extract dinov2-small at 112×112
# =============================================================================

SMALL_112_DIR="data/dinov2_features_small_112"

if [ ! -f "${SMALL_112_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting dinov2-small features at 112×112 ==="

    python scripts/extract_dinov2_features.py \
        --labels-file "${LABELS_CSV}" \
        --output-dir "${SMALL_112_DIR}" \
        --backbone dinov2-small \
        --sample-fps 12 \
        --window-seconds 1.0 \
        --frame-size 112 \
        --pool-mode mean_patch \
        --device auto

    echo "dinov2-small 112×112 extraction complete."
else
    echo "=== dinov2-small 112×112 features already extracted ==="
fi

# =============================================================================
# Extract dinov2-small at 168×168
# =============================================================================

SMALL_168_DIR="data/dinov2_features_small_168"

if [ ! -f "${SMALL_168_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting dinov2-small features at 168×168 ==="

    python scripts/extract_dinov2_features.py \
        --labels-file "${LABELS_CSV}" \
        --output-dir "${SMALL_168_DIR}" \
        --backbone dinov2-small \
        --sample-fps 12 \
        --window-seconds 1.0 \
        --frame-size 168 \
        --pool-mode mean_patch \
        --device auto

    echo "dinov2-small 168×168 extraction complete."
else
    echo "=== dinov2-small 168×168 features already extracted ==="
fi

echo ""
echo "=== Multi-resolution extraction complete ==="
echo "  112×112: ${SMALL_112_DIR}"
echo "  168×168: ${SMALL_168_DIR}"
echo "  224×224: data/dinov2_features_small_meanpatch (existing)"
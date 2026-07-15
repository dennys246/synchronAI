#!/bin/bash
SCRIPT_VERSION="pre_fnirs_extract_features_tile_bsub-v1"
# =============================================================================
# fNIRS Per-Pair Feature Extraction — TILE re-run (Option B)
#
# Extracts encoder features from the tiled-windowing generative models
# (runs/fnirs_perpair_{size}_tile) into fresh feature dirs
# (data/fnirs_perpair_{size}_tile_features). Mirrors
# pre_fnirs_extract_features_bsub.sh but with _tile paths, so the existing
# encoders/features stay untouched until we swap.
#
# The _tile diffusion config has target_len=469, and --stride-seconds 60 windows
# identically to the cache the models trained on — so features line up exactly
# with the training windows.
#
# Run AFTER: the four runs/fnirs_perpair_{size}_tile/ models finish.
# Run BEFORE: the tile classifier sweep.
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

# Feature variant: bottleneck (default) or multiscale (FEATURE_VARIANT=multiscale).
VARIANT_SUFFIX=""
MS_FLAG=""
if [ "${FEATURE_VARIANT:-bottleneck}" = "multiscale" ]; then
    VARIANT_SUFFIX="_multiscale"
    MS_FLAG="--multiscale"
fi

echo "=========================================="
echo "  [$SCRIPT_VERSION]  fNIRS Per-Pair Feature Extraction (TILE${VARIANT_SUFFIX})"
echo "  Date: $DATE"
echo "=========================================="

submit_extraction() {
    local MODEL_NAME="$1"
    local PRETRAIN_DIR="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}_tile"
    local CONFIG_JSON="${PRETRAIN_DIR}/fnirs_diffusion_config.json"
    local WEIGHTS_H5="${PRETRAIN_DIR}/fnirs_unet.weights.h5"
    local ENCODER_PT="${PRETRAIN_DIR}/fnirs_unet_encoder.pt"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_tile${VARIANT_SUFFIX}_features"

    echo ""
    echo "=== $MODEL_NAME (tile) ==="

    bsub -J "synchronai-extract-tile${VARIANT_SUFFIX}-${MODEL_NAME}-$DATE" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 24000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 16 \
         -R 'select[mem>24GB] rusage[mem=24GB] span[hosts=1]' \
         -oo "$LOG_DIR/fnirs_extract_tile${VARIANT_SUFFIX}_${MODEL_NAME}_$DATE.log" \
         -g /$USER/fnirs_extract_tile \
         << EXTRACT_EOF
echo "=== [$SCRIPT_VERSION] extract tile $MODEL_NAME ==="
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"
# Absolute ml-env python (source activate is unreliable in LSF heredocs) + block
# ~/.local so it can't pull a broken py3.12 hrfunc.
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"
export PYTHONNOUSERSITE=1
# Heavy per-item encoder forward — pin slots to one host so threads stay local.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

if [ ! -f "$ENCODER_PT" ]; then
    if [ ! -f "$WEIGHTS_H5" ]; then
        echo "ERROR: Pretrained weights not found: $WEIGHTS_H5 (has the tile run finished?)"
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

echo "=== Extracting per-pair features (tile) ==="
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
    $MS_FLAG
EXTRACT_STATUS=\$?

echo "=== [$SCRIPT_VERSION] extract tile $MODEL_NAME exited code=\$EXTRACT_STATUS ==="
if [ \$EXTRACT_STATUS -ne 0 ]; then exit \$EXTRACT_STATUS; fi
if [ -f "$FEATURE_DIR/feature_index.csv" ]; then
    echo "Feature count:"; wc -l "$FEATURE_DIR/feature_index.csv"
else
    echo "ERROR: No feature_index.csv produced!"; exit 1
fi
EXTRACT_EOF
}

submit_extraction "micro"
submit_extraction "small"
submit_extraction "medium"
submit_extraction "large"

echo ""
echo "=========================================="
echo "  4 tile extraction jobs submitted"
echo "  Monitor: bjobs -g /\$USER/fnirs_extract_tile"
echo "  Outputs: data/fnirs_perpair_{size}_tile_features/"
echo "=========================================="

#!/bin/bash
SCRIPT_VERSION="pre_fnirs_extract_random_tile_bsub-v1"
# =============================================================================
# fNIRS Random Encoder Feature Extraction — TILE re-run (for ablation)
#
# Randomly-initialized encoder (same _tile architecture, no pretrained weights)
# -> data/fnirs_perpair_{MODEL}_tile_random_features. The encoder .pt is read
# only for its architecture; --random-init ignores the weights.
#
# Submit one size at a time:  PERPAIR_MODEL=large bash <this>
# =============================================================================

MODEL_NAME="${PERPAIR_MODEL:-large}"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}_tile/fnirs_unet_encoder.pt"
FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_tile_random_features"

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

echo "=== [$SCRIPT_VERSION] random (tile) encoder extraction: $MODEL_NAME ==="

bsub -J "synchronai-extract-random-tile-${MODEL_NAME}-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 24000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 16 \
     -R 'select[mem>24GB] rusage[mem=24GB] span[hosts=1]' \
     -oo "$LOG_DIR/fnirs_extract_random_tile_${MODEL_NAME}_$DATE.log" \
     -g /$USER/fnirs_extract_tile \
     << EXTRACT_EOF
echo "=== [$SCRIPT_VERSION] random tile extract $MODEL_NAME ==="
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

if [ ! -f "$ENCODER_PT" ]; then
    echo "ERROR: encoder not found: $ENCODER_PT (has the tile run + its TF->PT conversion finished?)"
    exit 1
fi

rm -rf "$FEATURE_DIR"

"\$ML_PY" scripts/extract_fnirs_features.py \
    --encoder-weights "$ENCODER_PT" \
    --data-dirs "$FNIRS_DIRS" \
    --output-dir "$FEATURE_DIR" \
    --per-pair \
    --stride-seconds 60.0 \
    --random-init \
    --qc-cache "$SYNCHRONAI_DIR/data/qc_tiers.csv" \
    --include-tiers "gold,standard,salvageable" \
    --encoder-batch-size 32 \
    --pack-output \
    --delete-unpacked
EXTRACT_STATUS=\$?
echo "=== [$SCRIPT_VERSION] random tile extract $MODEL_NAME exited code=\$EXTRACT_STATUS ==="
if [ \$EXTRACT_STATUS -ne 0 ]; then exit \$EXTRACT_STATUS; fi
if [ -f "$FEATURE_DIR/feature_index.csv" ]; then
    wc -l "$FEATURE_DIR/feature_index.csv"
else
    echo "ERROR: No feature_index.csv produced!"; exit 1
fi
EXTRACT_EOF

echo "  Monitor: bjobs -g /\$USER/fnirs_extract_tile"

#!/bin/bash
SCRIPT_VERSION="pre_fnirs_extract_random_bsub-v1"
# =============================================================================
# fNIRS Random Encoder Feature Extraction (for ablation)
#
# Extracts features using randomly initialized encoder (same architecture,
# no pretrained weights). For proving pretraining is necessary.
# =============================================================================

MODEL_NAME="${PERPAIR_MODEL:-large}"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}/fnirs_unet_encoder.pt"
FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_random_features"

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

echo "=== [$SCRIPT_VERSION] ==="
echo "  Random encoder extraction: $MODEL_NAME"

bsub -J "synchronai-extract-random-${MODEL_NAME}-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 16000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 8 \
     -R 'select[mem>16GB] rusage[mem=16GB]' \
     -oo "$LOG_DIR/fnirs_extract_random_${MODEL_NAME}_$DATE.log" \
     -g /$USER/fnirs_extract \
     << EXTRACT_EOF
echo "=== [$SCRIPT_VERSION] random extract $MODEL_NAME ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"

rm -rf "$FEATURE_DIR"

python scripts/extract_fnirs_features.py \
    --encoder-weights "$ENCODER_PT" \
    --data-dirs "$FNIRS_DIRS" \
    --output-dir "$FEATURE_DIR" \
    --per-pair \
    --stride-seconds 60.0 \
    --random-init \
    --qc-cache "$SYNCHRONAI_DIR/data/qc_tiers.csv" \
    --include-tiers "gold,standard,salvageable" \
    --encoder-batch-size 32

echo "=== Random extraction complete ==="
if [ -f "$FEATURE_DIR/feature_index.csv" ]; then
    wc -l "$FEATURE_DIR/feature_index.csv"
else
    echo "ERROR: No feature_index.csv produced!"
    exit 1
fi
EXTRACT_EOF

echo "  Monitor: bjobs -g /\$USER/fnirs_extract"

#!/bin/bash
SCRIPT_VERSION="pre_fnirs_ablation_random_bsub-v5"
# =============================================================================
# fNIRS Ablation: Random Per-Pair Encoder vs Pretrained
#
# Proves that diffusion pretraining is essential. Extracts per-pair features
# from a randomly initialized U-Net encoder and trains lstm64 classifier.
#
# Single job: extract + train in same container (avoids NFS caching issues).
#
# Usage:
#   PERPAIR_MODEL=large sh scripts/bsub/pre_fnirs_ablation_random_bsub.sh
# =============================================================================

MODEL_NAME="${PERPAIR_MODEL:-large}"

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export ABLATION_MODEL_NAME="$MODEL_NAME"

export DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}/fnirs_unet_encoder.pt"

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

echo "=== [$SCRIPT_VERSION] ==="
echo "=========================================="
echo "  fNIRS Ablation: Random vs Pretrained"
echo "  Per-pair model: $MODEL_NAME"
echo "  Date: $DATE"
echo "=========================================="

echo ""
echo "Submitting combined extract+train job..."

bsub -J "synchronai-fnirs-random-${MODEL_NAME}-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 16000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 8 \
     -R 'select[mem>16GB] rusage[mem=16GB]' \
     -oo "$LOG_DIR/fnirs_ablation_random_${MODEL_NAME}_$DATE.log" \
     -g /$USER/fnirs_ablation \
     << 'ABLATION_EOF'
echo "=== [pre_fnirs_ablation_random_bsub-v5] extract+train ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_perpair_${ABLATION_MODEL_NAME}/fnirs_unet_encoder.pt"
RANDOM_FEATURES="$SYNCHRONAI_DIR/data/fnirs_perpair_${ABLATION_MODEL_NAME}_random_features"
ABLATION_DIR="$SYNCHRONAI_DIR/runs/fnirs_ablation_random"

if [ ! -f "$ENCODER_PT" ]; then
    echo "ERROR: Encoder .pt not found: $ENCODER_PT"
    echo "Run pre_fnirs_perpair_pretrain_bsub.sh and convert weights first."
    exit 1
fi

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

# --- Step 1: Extract with random encoder ---
if [ ! -f "$RANDOM_FEATURES/feature_index.csv" ]; then
    echo "=== Extracting per-pair features with RANDOM encoder ==="
    python scripts/extract_fnirs_features.py \
        --encoder-weights "$ENCODER_PT" \
        --data-dirs "$FNIRS_DIRS" \
        --output-dir "$RANDOM_FEATURES" \
        --per-pair \
        --stride-seconds 60.0 \
        --random-init \
        --enable-qc \
        --sci-threshold 0.40 \
        --snr-threshold 2.0 \
        --cardiac-peak-ratio 2.0 \
        --no-require-cardiac \
        --include-tiers "gold,standard,salvageable"
    if [ $? -ne 0 ] || [ ! -f "$RANDOM_FEATURES/feature_index.csv" ]; then
        echo "ERROR: Random feature extraction failed!"
        exit 1
    fi
else
    echo "=== Random features already extracted ==="
fi

# --- Step 2: Train lstm64 ---
echo ""
echo "=== Training ${ABLATION_MODEL_NAME}_lstm64 on random features ==="
python scripts/train_fnirs_from_features.py \
    --feature-dir "$RANDOM_FEATURES" \
    --save-dir "$ABLATION_DIR/${ABLATION_MODEL_NAME}_lstm64" \
    --label-column participant_type \
    --label-map "child:0,adult:1" \
    --hidden-dim 64 \
    --dropout 0.3 \
    --pool lstm \
    --learning-rate 3e-4 \
    --weight-decay 1e-2 \
    --warmup-epochs 3 \
    --patience 15 \
    --epochs 50 \
    --batch-size 32 \
    --num-workers 0 \
    --seed 42 \
    --include-tiers "gold,standard" \
    --holdout-tiers "gold,salvageable"

echo ""
echo "=== Ablation complete ==="
ABLATION_EOF

echo ""
echo "=========================================="
echo "  1 job submitted (extract + train in same container)"
echo "  Per-pair model: $MODEL_NAME (random init)"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_ablation"
echo "=========================================="

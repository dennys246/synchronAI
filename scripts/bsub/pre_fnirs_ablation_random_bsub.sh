#!/bin/sh
SCRIPT_VERSION="pre_fnirs_ablation_random_bsub-v4"
# =============================================================================
# fNIRS Ablation: Random Per-Pair Encoder vs Pretrained
#
# Proves that diffusion pretraining is essential for the child/adult classifier.
# Extracts per-pair features from a randomly initialized U-Net encoder (same
# architecture, no pretrained weights) and runs the best classifier (lstm64).
#
# Runs AFTER per-pair pretraining (needs the encoder .pt file for architecture
# config, but uses --random-init to ignore pretrained weights).
#
# Expected result: AUC ~0.5 (random chance) vs pretrained per-pair AUC.
#
# Usage:
#   # Default: ablate against "large" per-pair model
#   sh scripts/bsub/pre_fnirs_ablation_random_bsub.sh
#
#   # Ablate against a different size:
#   PERPAIR_MODEL=medium sh scripts/bsub/pre_fnirs_ablation_random_bsub.sh
# =============================================================================

MODEL_NAME="${PERPAIR_MODEL:-large}"

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/src:/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

# Needs the pretrained .pt for architecture config (but --random-init ignores weights)
ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}/fnirs_unet_encoder.pt"
RANDOM_FEATURES="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_random_features"
ABLATION_DIR="$SYNCHRONAI_DIR/runs/fnirs_ablation_random"

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

# =============================================================================
# Step 1: Extract features with random per-pair encoder
# =============================================================================

EXTRACT_JOB="synchronai-fnirs-random-extract-${MODEL_NAME}-$DATE"

# Export MODEL_NAME so the quoted heredoc can access it at runtime
export ABLATION_MODEL_NAME="$MODEL_NAME"

echo ""
echo "Submitting random extraction job..."

EXTRACT_OUTPUT=$(bsub -J "$EXTRACT_JOB" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 16000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 8 \
     -R 'select[mem>16GB] rusage[mem=16GB]' \
     -oo "$LOG_DIR/fnirs_random_extract_${MODEL_NAME}_$DATE.log" \
     -g /$USER/fnirs_ablation \
     << 'EXTRACT_EOF'
echo "=== [pre_fnirs_ablation_random_bsub-v4] ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_perpair_${ABLATION_MODEL_NAME}/fnirs_unet_encoder.pt"
RANDOM_FEATURES="$SYNCHRONAI_DIR/data/fnirs_perpair_${ABLATION_MODEL_NAME}_random_features"

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

echo "Random extraction complete."
EXTRACT_EOF
)
echo "$EXTRACT_OUTPUT"
EXTRACT_JOBID=$(echo "$EXTRACT_OUTPUT" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*')
echo "  Extract job: $EXTRACT_JOB (ID: $EXTRACT_JOBID)"

# =============================================================================
# Step 2: Train lstm64 on random features (same config as best classifier)
# =============================================================================

echo ""
echo "Submitting training job (waits for extraction)..."

bsub -J "synchronai-fnirs-random-lstm64-${MODEL_NAME}-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 4000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 4 \
     -R 'select[mem>4GB] rusage[mem=4GB]' \
     -w "done($EXTRACT_JOBID)" \
     -oo "$LOG_DIR/fnirs_ablation_random_lstm64_${MODEL_NAME}_$DATE.log" \
     -g /$USER/fnirs_ablation \
     << 'TRAIN_EOF'
echo "=== [pre_fnirs_ablation_random_bsub-v4] train ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

# Force NFS metadata cache refresh and verify features exist
FEAT_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${ABLATION_MODEL_NAME}_random_features"
ls "$FEAT_DIR/feature_index.csv" > /dev/null 2>&1
if [ ! -f "$FEAT_DIR/feature_index.csv" ]; then
    echo "ERROR: feature_index.csv not found at $FEAT_DIR"
    echo "Waiting 30s for NFS cache refresh..."
    sleep 30
    ls "$FEAT_DIR/" > /dev/null 2>&1
    if [ ! -f "$FEAT_DIR/feature_index.csv" ]; then
        echo "ERROR: Still not found after retry. Setup may have failed."
        exit 1
    fi
fi
echo "Found feature_index.csv at $FEAT_DIR"

python scripts/train_fnirs_from_features.py \
    --feature-dir "$SYNCHRONAI_DIR/data/fnirs_perpair_${ABLATION_MODEL_NAME}_random_features" \
    --save-dir "$SYNCHRONAI_DIR/runs/fnirs_ablation_random/${ABLATION_MODEL_NAME}_lstm64" \
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
TRAIN_EOF

echo "  Training job: synchronai-fnirs-random-lstm64-${MODEL_NAME}-$DATE"

echo ""
echo "=========================================="
echo "  1 extraction + 1 training job submitted"
echo "  Per-pair model: $MODEL_NAME (random init)"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_ablation"
echo ""
echo "  Compare results:"
echo "    Pretrained: runs/fnirs_child_adult_sweep/${MODEL_NAME}_lstm64"
echo "    Random:     $ABLATION_DIR/${MODEL_NAME}_lstm64"
echo "=========================================="

#!/bin/sh
# =============================================================================
# fNIRS Ablation: Random Encoder vs Pretrained
#
# Proves that the diffusion pretraining is essential for the child/adult
# classifier. Extracts features from a randomly initialized U-Net encoder
# (same architecture, no pretrained weights) and runs the best model
# (bn_lstm64) on those features.
#
# Expected result: AUC ~0.5 (random chance) vs 0.97 with pretraining.
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt"
RANDOM_FEATURES="data/fnirs_random_encoder_features"
ABLATION_DIR="runs/fnirs_ablation_random"

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

echo "=========================================="
echo "  fNIRS Ablation: Random vs Pretrained"
echo "  Date: $DATE"
echo "=========================================="

# =============================================================================
# Step 1: Extract features with random encoder
# =============================================================================

EXTRACT_JOB="synchronai-fnirs-random-extract-$DATE"

echo ""
echo "Submitting random extraction job..."

bsub -J "$EXTRACT_JOB" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 16000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 8 \
     -R 'select[mem>16GB] rusage[mem=16GB]' \
     -oo "$LOG_DIR/fnirs_random_extract_$DATE.log" \
     -g /$USER/fnirs_ablation \
     << 'EXTRACT_EOF'
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
pip install -e . 2>/dev/null

ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt"
RANDOM_FEATURES="data/fnirs_random_encoder_features"

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

echo "=== Extracting features with RANDOM encoder ==="
python scripts/extract_fnirs_features.py \
    --encoder-weights "$ENCODER_PT" \
    --data-dirs "$FNIRS_DIRS" \
    --output-dir "$RANDOM_FEATURES" \
    --stride-seconds 60.0 \
    --random-init

echo "Random extraction complete."
EXTRACT_EOF

echo "  Extract job: $EXTRACT_JOB"

# =============================================================================
# Step 2: Train bn_lstm64 on random features (same config as winning run)
# =============================================================================

echo ""
echo "Submitting training job (waits for extraction)..."

bsub -J "synchronai-fnirs-random-lstm64-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 4000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 4 \
     -R 'select[mem>4GB] rusage[mem=4GB]' \
     -w "done($EXTRACT_JOB)" \
     -oo "$LOG_DIR/fnirs_ablation_random_lstm64_$DATE.log" \
     -g /$USER/fnirs_ablation \
     << 'TRAIN_EOF'
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
pip install -e . 2>/dev/null

python scripts/train_fnirs_from_features.py \
    --feature-dir "data/fnirs_random_encoder_features" \
    --save-dir "runs/fnirs_ablation_random/bn_lstm64" \
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
    --seed 42
TRAIN_EOF

echo "  Training job: synchronai-fnirs-random-lstm64-$DATE"

echo ""
echo "=========================================="
echo "  1 extraction + 1 training job submitted"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_ablation"
echo ""
echo "  Compare results:"
echo "    Pretrained: runs/fnirs_child_adult_sweep/bn_lstm64/history.json"
echo "    Random:     runs/fnirs_ablation_random/bn_lstm64/history.json"
echo "=========================================="

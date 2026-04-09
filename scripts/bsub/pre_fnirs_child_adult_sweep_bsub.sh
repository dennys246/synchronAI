#!/bin/bash
SCRIPT_VERSION="pre_fnirs_child_adult_sweep-v6"
# =============================================================================
# fNIRS Child/Adult Classification Sweep — Per-Pair Architecture
#
# For each pretrained per-pair model (micro/small/medium/large):
#   1. Convert TF weights to PyTorch (if not done)
#   2. Extract per-pair features with relaxed QC (all tiers)
#   3. Train 5 classifier architectures with holdout-tier evaluation
#
# All steps run in a SINGLE job per model to avoid NFS caching issues
# between Docker containers. Each job does extract + 5 classifiers.
#
# Total: 4 jobs (1 per model size, each runs extract + 5 classifiers)
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"

export DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

# Data directories
FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

SWEEP_DIR="$SYNCHRONAI_DIR/runs/fnirs_child_adult_sweep"

echo "=========================================="
echo "  [$SCRIPT_VERSION]"
echo "  fNIRS Child/Adult Sweep (Per-Pair)"
echo "  Date: $DATE"
echo "=========================================="

# =============================================================================
# Submit one job per model size (extract + train all classifiers in same container)
# =============================================================================

submit_model_sweep() {
    local MODEL_NAME="$1"

    local PRETRAIN_DIR="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}"
    local CONFIG_JSON="${PRETRAIN_DIR}/fnirs_diffusion_config.json"
    local WEIGHTS_H5="${PRETRAIN_DIR}/fnirs_unet.weights.h5"
    local ENCODER_PT="${PRETRAIN_DIR}/fnirs_unet_encoder.pt"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_features"

    echo ""
    echo "=== $MODEL_NAME ==="
    echo "  Submitting combined extract+train job..."

    bsub -J "synchronai-sweep-${MODEL_NAME}-$DATE" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 16000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 8 \
         -R 'select[mem>16GB] rusage[mem=16GB]' \
         -oo "$LOG_DIR/fnirs_sweep_${MODEL_NAME}_$DATE.log" \
         -g /$USER/fnirs_sweep \
         << SWEEP_EOF
echo "=== [$SCRIPT_VERSION] $MODEL_NAME extract+train ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"

# --- Step 1: Convert TF weights to PyTorch (if needed) ---
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
else
    echo "=== PyTorch encoder already exists ==="
fi

# --- Step 2: Extract per-pair features (if needed) ---
if [ ! -f "$FEATURE_DIR/feature_index.csv" ]; then
    echo "=== Extracting per-pair features (all tiers) ==="
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
    if [ \$? -ne 0 ] || [ ! -f "$FEATURE_DIR/feature_index.csv" ]; then
        echo "ERROR: Feature extraction failed!"
        exit 1
    fi
else
    echo "=== Features already extracted ==="
fi

echo "=== Feature index: \$(wc -l < "$FEATURE_DIR/feature_index.csv") lines ==="

# --- Step 3: Train all 5 classifiers sequentially ---
train_classifier() {
    local RUN_NAME="\$1"
    local HIDDEN_DIM="\$2"
    local DROPOUT="\$3"
    local POOL="\$4"
    local LR="\$5"

    echo ""
    echo "=== Training ${MODEL_NAME}_\${RUN_NAME} (h=\${HIDDEN_DIM}, pool=\${POOL}) ==="

    python scripts/train_fnirs_from_features.py \
        --feature-dir "$FEATURE_DIR" \
        --save-dir "$SWEEP_DIR/${MODEL_NAME}_\${RUN_NAME}" \
        --label-column participant_type \
        --label-map "child:0,adult:1" \
        --hidden-dim \$HIDDEN_DIM \
        --dropout \$DROPOUT \
        --pool \$POOL \
        --learning-rate \$LR \
        --weight-decay 1e-2 \
        --warmup-epochs 3 \
        --patience 15 \
        --epochs 50 \
        --batch-size 32 \
        --num-workers 0 \
        --seed 42 \
        --include-tiers "gold,standard" \
        --holdout-tiers "gold,salvageable"
}

train_classifier "linear"     0   0.0 "mean" "1e-3"
train_classifier "mlp32"      32  0.3 "mean" "3e-4"
train_classifier "mlp64_proj" 64  0.5 "mean" "3e-4"
train_classifier "lstm64"     64  0.3 "lstm" "3e-4"
train_classifier "lstm_proj"  64  0.5 "lstm" "3e-4"

echo ""
echo "=== All classifiers complete for $MODEL_NAME ==="
SWEEP_EOF
}

# Submit for all 4 per-pair model sizes (run in parallel)
submit_model_sweep "micro"
submit_model_sweep "small"
submit_model_sweep "medium"
submit_model_sweep "large"

echo ""
echo "=========================================="
echo "  4 jobs submitted (each runs extract + 5 classifiers)"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_sweep"
echo ""
echo "  After all complete:"
echo "    Compare across model sizes and classifiers"
echo "    Previous 20-ch baseline: AUC 0.974 (bn_lstm64)"
echo "=========================================="

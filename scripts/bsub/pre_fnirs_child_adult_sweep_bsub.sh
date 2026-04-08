#!/bin/bash
SCRIPT_VERSION="pre_fnirs_child_adult_sweep-v4"
# =============================================================================
# fNIRS Child/Adult Classification Sweep — Per-Pair Architecture
#
# Runs AFTER per-pair pretraining (pre_fnirs_perpair_pretrain_bsub.sh).
#
# For each pretrained per-pair model (micro/small/medium/large):
#   Step 1: Convert TF weights to PyTorch (if not done)
#   Step 2: Extract per-pair features with relaxed QC (all tiers)
#   Step 3: Train 5 classifier architectures with holdout-tier evaluation
#
# Sweep variants per model size:
#   linear:     Linear probe (mean pool, hidden=0)
#   mlp32:      Small MLP (mean pool, hidden=32)
#   mlp64_proj: MLP with projection (mean pool, hidden=64, dropout=0.5)
#   lstm64:     LSTM temporal model (hidden=64)
#   lstm_proj:  LSTM with projection (hidden=64, dropout=0.5)
#
# Total: 4 setup + 20 training jobs (4 models × 5 classifiers)
#
# All training jobs include --holdout-tiers "gold,salvageable" for
# per-epoch evaluation on pristine and high-motion val subsets.
#
# Usage:
#   sh scripts/bsub/pre_fnirs_child_adult_sweep_bsub.sh
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/src:/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

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
# Per-model pipeline: convert + extract + sweep classifiers
# =============================================================================

submit_model_sweep() {
    local MODEL_NAME="$1"  # small, medium, large

    local PRETRAIN_DIR="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}"
    local CONFIG_JSON="${PRETRAIN_DIR}/fnirs_diffusion_config.json"
    local WEIGHTS_H5="${PRETRAIN_DIR}/fnirs_unet.weights.h5"
    local ENCODER_PT="${PRETRAIN_DIR}/fnirs_unet_encoder.pt"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_features"

    # --- Setup job: convert + extract ---
    local SETUP_JOB="synchronai-sweep-setup-${MODEL_NAME}-$DATE"

    echo ""
    echo "=== $MODEL_NAME ==="
    echo "  Submitting setup (convert + extract)..."

    bsub -J "$SETUP_JOB" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 16000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 8 \
         -R 'select[mem>16GB] rusage[mem=16GB]' \
         -oo "$LOG_DIR/fnirs_sweep_setup_${MODEL_NAME}_$DATE.log" \
         -g /$USER/fnirs_sweep \
         << SETUP_EOF > /tmp/bsub_setup_${MODEL_NAME}_$$.out 2>&1
echo "=== [$SCRIPT_VERSION] setup $MODEL_NAME ==="
cd \$SYNCHRONAI_DIR
. "\$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

CONFIG_JSON="$CONFIG_JSON"
WEIGHTS_H5="$WEIGHTS_H5"
ENCODER_PT="$ENCODER_PT"
FEATURE_DIR="$FEATURE_DIR"

if [ ! -f "\$WEIGHTS_H5" ]; then
    echo "ERROR: Pretrained weights not found: \$WEIGHTS_H5"
    echo "Run pre_fnirs_perpair_pretrain_bsub.sh first."
    exit 1
fi

# Convert TF -> PyTorch
if [ ! -f "\$ENCODER_PT" ]; then
    echo "=== Converting TF weights to PyTorch ==="
    python scripts/convert_fnirs_tf_to_pt.py \
        --config-json "\$CONFIG_JSON" \
        --weights-path "\$WEIGHTS_H5" \
        --output "\$ENCODER_PT" \
        --verify
    if [ \$? -ne 0 ]; then
        echo "ERROR: Weight conversion failed!"
        exit 1
    fi
else
    echo "=== PyTorch encoder already exists ==="
fi

FNIRS_DIRS="$FNIRS_DIRS"

# Extract per-pair features with relaxed QC (captures all tiers)
if [ ! -f "\${FEATURE_DIR}/feature_index.csv" ]; then
    echo "=== Extracting per-pair features (all tiers) ==="
    python scripts/extract_fnirs_features.py \
        --encoder-weights "\$ENCODER_PT" \
        --data-dirs "\$FNIRS_DIRS" \
        --output-dir "\$FEATURE_DIR" \
        --per-pair \
        --stride-seconds 60.0 \
        --enable-qc \
        --sci-threshold 0.40 \
        --snr-threshold 2.0 \
        --cardiac-peak-ratio 2.0 \
        --no-require-cardiac \
        --include-tiers "gold,standard,salvageable"
else
    echo "=== Features already extracted ==="
fi

echo "=== Setup complete for $MODEL_NAME ==="
SETUP_EOF

    cat /tmp/bsub_setup_${MODEL_NAME}_$$.out
    local SETUP_JOBID
    SETUP_JOBID=$(grep -o 'Job <[0-9]*>' /tmp/bsub_setup_${MODEL_NAME}_$$.out | grep -o '[0-9]*')
    rm -f /tmp/bsub_setup_${MODEL_NAME}_$$.out
    echo "  Setup job: $SETUP_JOB (ID: $SETUP_JOBID)"

    # --- Classification sweep jobs ---
    echo "  Submitting 5 classifier jobs..."

    submit_classifier() {
        local RUN_NAME="$1"
        local HIDDEN_DIM="$2"
        local DROPOUT="$3"
        local POOL="$4"
        local LR="$5"

        echo "    ${MODEL_NAME}_${RUN_NAME} (h=$HIDDEN_DIM, pool=$POOL)"

        bsub -J "synchronai-sweep-${MODEL_NAME}-${RUN_NAME}-$DATE" \
             -G compute-perlmansusan \
             -q general \
             -m general \
             -M 4000000 \
             -a 'docker(continuumio/anaconda3)' \
             -n 4 \
             -R 'select[mem>4GB] rusage[mem=4GB]' \
             -w "done($SETUP_JOBID)" \
             -oo "$LOG_DIR/fnirs_sweep_${MODEL_NAME}_${RUN_NAME}_$DATE.log" \
             -g /$USER/fnirs_sweep \
             << EOF
echo "=== [$SCRIPT_VERSION] train ${MODEL_NAME}_${RUN_NAME} ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

# Force NFS metadata cache refresh and verify features exist
ls "$FEATURE_DIR/feature_index.csv" > /dev/null 2>&1
if [ ! -f "$FEATURE_DIR/feature_index.csv" ]; then
    echo "ERROR: feature_index.csv not found at $FEATURE_DIR"
    echo "Waiting 30s for NFS cache refresh..."
    sleep 30
    ls "$FEATURE_DIR/" > /dev/null 2>&1
    if [ ! -f "$FEATURE_DIR/feature_index.csv" ]; then
        echo "ERROR: Still not found after retry. Setup may have failed."
        exit 1
    fi
fi
echo "Found feature_index.csv: $(wc -l < "$FEATURE_DIR/feature_index.csv") lines"

python scripts/train_fnirs_from_features.py \
    --feature-dir "$FEATURE_DIR" \
    --save-dir "$SWEEP_DIR/${MODEL_NAME}_${RUN_NAME}" \
    --label-column participant_type \
    --label-map "child:0,adult:1" \
    --hidden-dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --pool $POOL \
    --learning-rate $LR \
    --weight-decay 1e-2 \
    --warmup-epochs 3 \
    --patience 15 \
    --epochs 50 \
    --batch-size 32 \
    --num-workers 0 \
    --seed 42 \
    --include-tiers "gold,standard" \
    --holdout-tiers "gold,salvageable"
EOF
    }

    submit_classifier "linear"     0   0.0 "mean" "1e-3"
    submit_classifier "mlp32"      32  0.3 "mean" "3e-4"
    submit_classifier "mlp64_proj" 64  0.5 "mean" "3e-4"
    submit_classifier "lstm64"     64  0.3 "lstm" "3e-4"
    submit_classifier "lstm_proj"  64  0.5 "lstm" "3e-4"
}

# Submit for all 4 per-pair model sizes
submit_model_sweep "micro"
submit_model_sweep "small"
submit_model_sweep "medium"
submit_model_sweep "large"

echo ""
echo "=========================================="
echo "  4 setup + 20 training jobs submitted"
echo "  Training jobs wait for their setup job"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_sweep"
echo ""
echo "  After all complete:"
echo "    Compare across model sizes and classifiers"
echo "    Previous 20-ch baseline: AUC 0.974 (bn_lstm64)"
echo "=========================================="

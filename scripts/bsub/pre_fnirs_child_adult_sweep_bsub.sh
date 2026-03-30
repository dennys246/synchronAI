#!/bin/sh
# =============================================================================
# fNIRS Child/Adult Classification Sweep — Full Pipeline
#
# Unified bsub script that runs the complete pipeline:
#   Step 1: Convert TF U-Net weights to PyTorch (one-time)
#   Step 2: Extract bottleneck features (non-overlapping 60s windows)
#   Step 2b: Extract multiscale features (non-overlapping 60s windows)
#   Step 3: Submit 11 sweep training jobs
#
# The child/adult classifier is NOT a sanity check — it's a standalone
# model whose learned representations transfer to synchrony prediction.
#
# After sweep completes:
#   python scripts/compare_sweep_results.py --sweep-dir runs/fnirs_child_adult_sweep
# =============================================================================

# Shared environment
export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI:$PYTHONPATH"

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"

export DATE=$(date +'%m-%d')
LOG_DIR="$SYNCHRONAI_DIR/scripts/bsub/logs"
mkdir -p "$LOG_DIR"

# Data directories (same as generative_pretrain.sh)
FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

# Paths
CONFIG_JSON="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_diffusion_config.json"
WEIGHTS_H5="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_unet.weights.h5"
ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt"

BOTTLENECK_DIR="data/fnirs_encoder_features"
MULTISCALE_DIR="data/fnirs_multiscale_features"
SWEEP_DIR="runs/fnirs_child_adult_sweep"

SWEEP_SCRIPT="$SYNCHRONAI_DIR/scripts/bsub/fnirs_child_adult_sweep_bsub.sh"

echo "=========================================="
echo "  fNIRS Child/Adult Classification Sweep"
echo "  Date: $DATE"
echo "=========================================="

# =============================================================================
# Step 1: Submit conversion + extraction job
# =============================================================================

SETUP_JOB="synchronai-fnirs-setup-$DATE"

echo ""
echo "Submitting setup job (convert + extract)..."

bsub -J "$SETUP_JOB" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 16000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 8 \
     -R 'select[mem>16GB] rusage[mem=16GB]' \
     -oo "$LOG_DIR/fnirs_setup_$DATE.log" \
     -g /$USER/fnirs_sweep \
     << 'SETUP_EOF'
cd $SYNCHRONAI_DIR

CONFIG_JSON="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_diffusion_config.json"
WEIGHTS_H5="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_unet.weights.h5"
ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt"

# =============================================================================
# Step 1: Convert TF weights to PyTorch
#
# Uses conda base environment (has TensorFlow pre-installed).
# Do NOT install TF into ml-env — it conflicts with PyTorch and can
# cause disk quota issues. The conversion script only needs TF to load
# the .h5 weights, then saves a pure PyTorch .pt file.
# =============================================================================

# Activate ml-env (needs h5py, torch, numpy — NO tensorflow)
if [ -d "$SYNCHRONAI_DIR/ml-env" ]; then
    . "$SYNCHRONAI_DIR/ml-env/bin/activate"
    pip install -e . 2>/dev/null
    pip install --no-cache-dir h5py 2>/dev/null
else
    echo "ERROR: ml-env not found"
    exit 1
fi

if [ ! -f "$ENCODER_PT" ]; then
    echo ""
    echo "=== Converting TF weights to PyTorch (via h5py, no TensorFlow needed) ==="

    python scripts/convert_fnirs_tf_to_pt.py \
        --config-json "$CONFIG_JSON" \
        --weights-path "$WEIGHTS_H5" \
        --output "$ENCODER_PT" \
        --verify

    if [ $? -ne 0 ]; then
        echo "ERROR: Weight conversion failed!"
        exit 1
    fi
    echo "Conversion complete."
else
    echo "=== PyTorch encoder already exists: $ENCODER_PT ==="
fi

# All fNIRS data directories
FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

# --- Step 2: Extract bottleneck features ---
BOTTLENECK_DIR="data/fnirs_encoder_features"
if [ ! -f "${BOTTLENECK_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting bottleneck features (non-overlapping 60s windows) ==="
    python scripts/extract_fnirs_features.py \
        --encoder-weights "$ENCODER_PT" \
        --data-dirs "$FNIRS_DIRS" \
        --output-dir "$BOTTLENECK_DIR" \
        --stride-seconds 60.0
else
    echo "=== Bottleneck features already extracted ==="
fi

# --- Step 2b: Extract multiscale features ---
MULTISCALE_DIR="data/fnirs_multiscale_features"
if [ ! -f "${MULTISCALE_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting multiscale features ==="
    python scripts/extract_fnirs_features.py \
        --encoder-weights "$ENCODER_PT" \
        --data-dirs "$FNIRS_DIRS" \
        --output-dir "$MULTISCALE_DIR" \
        --multiscale \
        --stride-seconds 60.0
else
    echo "=== Multiscale features already extracted ==="
fi

echo ""
echo "=== Setup complete ==="
SETUP_EOF

echo "  Setup job: $SETUP_JOB"

# =============================================================================
# Step 3: Submit sweep training jobs (wait for setup)
# =============================================================================

echo ""
echo "Submitting 11 training jobs (will wait for setup)..."

# Helper function to submit a sweep job
submit_sweep() {
    local RUN_NAME="$1"
    local FEATURE_DIR="$2"
    local HIDDEN_DIM="$3"
    local DROPOUT="$4"
    local POOL="$5"
    local LR="$6"

    echo "  $RUN_NAME (h=$HIDDEN_DIM, d=$DROPOUT, pool=$POOL)"

    bsub -J "synchronai-fnirs-${RUN_NAME}-$DATE" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 4000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 4 \
         -R 'select[mem>4GB] rusage[mem=4GB]' \
         -w "done($SETUP_JOB)" \
         -oo "$LOG_DIR/fnirs_sweep_${RUN_NAME}_$DATE.log" \
         -g /$USER/fnirs_sweep \
         << EOF
cd \$SYNCHRONAI_DIR
. "\$SYNCHRONAI_DIR/ml-env/bin/activate"
pip install -e . 2>/dev/null

python scripts/train_fnirs_from_features.py \
    --feature-dir "$FEATURE_DIR" \
    --save-dir "$SWEEP_DIR/$RUN_NAME" \
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
    --seed 42
EOF
}

# --- GROUP A: Bottleneck features (512-dim, 59 timesteps) ---

submit_sweep "bn_linear"     "$BOTTLENECK_DIR" 0   0.0 "mean" "1e-3"
submit_sweep "bn_mlp32"      "$BOTTLENECK_DIR" 32  0.3 "mean" "3e-4"
submit_sweep "bn_mlp64_proj" "$BOTTLENECK_DIR" 64  0.5 "mean" "3e-4"
submit_sweep "bn_lstm64"     "$BOTTLENECK_DIR" 64  0.3 "lstm" "3e-4"
submit_sweep "bn_lstm_proj"  "$BOTTLENECK_DIR" 64  0.5 "lstm" "3e-4"

# --- GROUP B: Multiscale features (960-dim, mean-pooled) ---

submit_sweep "ms_linear"     "$MULTISCALE_DIR" 0   0.0 "mean" "1e-3"
submit_sweep "ms_mlp32"      "$MULTISCALE_DIR" 32  0.3 "mean" "3e-4"
submit_sweep "ms_mlp64_proj"  "$MULTISCALE_DIR" 64  0.5 "mean" "3e-4"
submit_sweep "ms_mlp64_hvreg" "$MULTISCALE_DIR" 64  0.7 "mean" "1e-4"
submit_sweep "ms_mlp128"     "$MULTISCALE_DIR" 128 0.5 "mean" "3e-4"

# --- GROUP C: Overlap variant (best arch re-tested with 30s stride) ---
# Note: This uses bottleneck features extracted with 30s stride overlap.
# The overlap extraction is done inline since it's a single variant.

submit_sweep "bn_mlp32_overlap" "$BOTTLENECK_DIR" 32 0.3 "mean" "3e-4"
# ^ This will use the non-overlapping features for now.
#   To test overlap properly, a separate extraction with --stride-seconds 30.0
#   would be needed. Add a manual step after initial results if warranted.

echo ""
echo "=========================================="
echo "  1 setup + 11 training jobs submitted"
echo "  Training jobs wait for setup to finish"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_sweep"
echo ""
echo "  After all complete:"
echo "    python scripts/compare_sweep_results.py --sweep-dir $SWEEP_DIR"
echo "=========================================="

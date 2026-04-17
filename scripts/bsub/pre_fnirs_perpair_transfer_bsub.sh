#!/bin/sh
SCRIPT_VERSION="pre_fnirs_perpair_transfer_bsub-v12"
PLOT_EVERY=3  # write history.png every N epochs during training
# =============================================================================
# fNIRS Per-Pair Transfer Learning: Classification Sweep
#
# Phase 2 of the per-pair pipeline. Runs AFTER pretraining completes.
#
# For each pretrained model (small/medium/large):
#   1. Convert TF weights to PyTorch encoder
#   2. Extract per-pair bottleneck features (with QC filtering)
#   3. Train child/adult classifiers (per-pair + aggregated)
#
# Quality control (matches pretraining pipeline):
#   - SCI >= 0.75, SNR >= 5.0, cardiac band verification
#   - Recordings that fail QC are excluded from feature extraction
#
# Per-pair classification: each pair classified independently, predictions
# averaged across pairs for a per-person prediction.
#
# Aggregated: all 10 pair features concatenated/pooled into one per-person
# vector, then classified. More comparable to 20-channel baseline.
#
# Compare results against 20-channel baseline (AUC 0.974 with bn_lstm64).
# =============================================================================

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

FNIRS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
FNIRS_DIRS="${FNIRS_DIRS}:/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"

SWEEP_DIR="$SYNCHRONAI_DIR/runs/fnirs_perpair_sweep"

echo "=== [$SCRIPT_VERSION] ==="
echo "=========================================="
echo "  fNIRS Per-Pair Transfer Learning Sweep"
echo "  Date: $DATE"
echo "=========================================="

# =============================================================================
# For each pretrained model: convert + extract + classify
# =============================================================================

submit_pipeline() {
    local MODEL_NAME="$1"       # small, medium, large
    local BASE_WIDTH="$2"       # for model config

    local PRETRAIN_DIR="$SYNCHRONAI_DIR/runs/fnirs_perpair_${MODEL_NAME}"
    local CONFIG_JSON="${PRETRAIN_DIR}/fnirs_diffusion_config.json"
    local WEIGHTS_H5="${PRETRAIN_DIR}/fnirs_unet.weights.h5"
    local ENCODER_PT="${PRETRAIN_DIR}/fnirs_unet_encoder.pt"
    local FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${MODEL_NAME}_features"

    # --- Step 1: Setup job (convert + extract) ---
    local SETUP_JOB="synchronai-perpair-setup-${MODEL_NAME}-$DATE"

    echo ""
    echo "=== $MODEL_NAME (base=$BASE_WIDTH) ==="
    echo "  Submitting setup (convert + extract)..."

    bsub -J "$SETUP_JOB" \
         -G compute-perlmansusan \
         -q general \
         -m general \
         -M 16000000 \
         -a 'docker(continuumio/anaconda3)' \
         -n 8 \
         -R 'select[mem>16GB] rusage[mem=16GB]' \
         -oo "$LOG_DIR/fnirs_perpair_setup_${MODEL_NAME}_$DATE.log" \
         -g /$USER/fnirs_perpair_transfer \
         << SETUP_EOF > /tmp/bsub_transfer_setup_${MODEL_NAME}_$$.out 2>&1
echo "=== [$SCRIPT_VERSION] ==="
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"
# Invoke ml-env's python by absolute path. Sourcing ml-env/bin/activate
# inside LSF heredocs does not reliably prepend ml-env/bin to PATH on
# this cluster — \`python\` silently resolves to the container's base
# /opt/conda/bin/python which does not have torch. See docs/troubleshooting.md.
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"
set -e

CONFIG_JSON="$CONFIG_JSON"
WEIGHTS_H5="$WEIGHTS_H5"
ENCODER_PT="$ENCODER_PT"
FEATURE_DIR="$FEATURE_DIR"

# Check pretrained model exists
if [ ! -f "\$WEIGHTS_H5" ]; then
    echo "ERROR: Pretrained weights not found: \$WEIGHTS_H5"
    echo "Run pre_fnirs_perpair_pretrain_bsub.sh first."
    exit 1
fi

# Convert TF -> PyTorch
if [ ! -f "\$ENCODER_PT" ]; then
    echo "=== Converting TF weights to PyTorch ==="
    "\$ML_PY" scripts/convert_fnirs_tf_to_pt.py \
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

# Extract per-pair features. Skip only when BOTH the index CSV and the
# packed mmap binary exist — otherwise we'd silently train on the slow
# unpacked path if someone resurrected an old index without packing.
if [ ! -f "\${FEATURE_DIR}/feature_index.csv" ] || [ ! -f "\${FEATURE_DIR}/features_packed.bin" ]; then
    echo "=== Extracting per-pair features ==="

    FNIRS_DIRS="$FNIRS_DIRS"

    "\$ML_PY" scripts/extract_fnirs_features.py \
        --encoder-weights "\$ENCODER_PT" \
        --data-dirs "\$FNIRS_DIRS" \
        --output-dir "\$FEATURE_DIR" \
        --stride-seconds 60.0 \
        --qc-cache "$SYNCHRONAI_DIR/data/qc_tiers.csv" \
        --include-tiers "gold,standard,salvageable" \
        --encoder-batch-size 32 \
        --pack-output \
        --delete-unpacked
    extract_rc=\$?
    if [ \$extract_rc -ne 0 ]; then
        echo "ERROR: extraction exited with code \$extract_rc"
        exit \$extract_rc
    fi
else
    echo "=== Features already extracted ==="
fi

# Post-setup verification: both the index CSV and the packed binary must
# exist before we declare success. Prevents downstream training jobs from
# running on stale unpacked data when extraction silently no-ops.
if [ ! -f "\${FEATURE_DIR}/feature_index.csv" ]; then
    echo "ERROR: feature_index.csv missing after setup — extraction failed silently"
    exit 1
fi
if [ ! -f "\${FEATURE_DIR}/features_packed.bin" ]; then
    echo "ERROR: features_packed.bin missing after setup — pack step did not run"
    exit 1
fi
if [ ! -f "\${FEATURE_DIR}/features_meta.json" ]; then
    echo "ERROR: features_meta.json missing after setup — pack step incomplete"
    exit 1
fi
echo "Verified: feature_index.csv + features_packed.bin + features_meta.json all present"

echo "=== Setup complete for $MODEL_NAME ==="
SETUP_EOF

    cat /tmp/bsub_transfer_setup_${MODEL_NAME}_$$.out
    local SETUP_JOBID
    SETUP_JOBID=$(grep -o 'Job <[0-9]*>' /tmp/bsub_transfer_setup_${MODEL_NAME}_$$.out | grep -o '[0-9]*')
    rm -f /tmp/bsub_transfer_setup_${MODEL_NAME}_$$.out
    if [ -z "$SETUP_JOBID" ]; then
        echo "ERROR: Setup job submission failed for $MODEL_NAME"
        return 1
    fi
    echo "  Setup job: $SETUP_JOB (ID: $SETUP_JOBID)"

    # --- Step 2: Classification sweep jobs ---
    echo "  Submitting classification jobs..."

    submit_classifier() {
        local RUN_NAME="$1"
        local HIDDEN_DIM="$2"
        local DROPOUT="$3"
        local POOL="$4"
        local LR="$5"

        echo "    $RUN_NAME (h=$HIDDEN_DIM, pool=$POOL)"

        # 16GB RAM: packed features are loaded via np.memmap. The full array
        # can exceed RAM (large ~62GB) — the OS pages in on demand. 16GB is
        # enough working set for any model size; large will see some page
        # churn but is still dramatically faster than per-file loads.
        bsub -J "synchronai-perpair-${MODEL_NAME}-${RUN_NAME}-$DATE" \
             -G compute-perlmansusan \
             -q general \
             -m general \
             -M 16000000 \
             -a 'docker(continuumio/anaconda3)' \
             -n 4 \
             -R 'select[mem>16GB] rusage[mem=16GB]' \
             -w "done($SETUP_JOBID)" \
             -oo "$LOG_DIR/fnirs_perpair_${MODEL_NAME}_${RUN_NAME}_$DATE.log" \
             -g /$USER/fnirs_perpair_transfer \
             << EOF
echo "=== [$SCRIPT_VERSION] train ${MODEL_NAME}_${RUN_NAME} ==="
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

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
echo "Found feature_index.csv at $FEATURE_DIR"

"\$ML_PY" scripts/train_fnirs_from_features.py \
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
    --holdout-tiers "gold,salvageable" \
    --plot-every $PLOT_EVERY
EOF
    }

    # Per-pair classifiers (each pair classified independently)
    submit_classifier "lstm64"     64  0.3 "lstm" "3e-4"
    submit_classifier "mlp32"      32  0.3 "mean" "3e-4"
    submit_classifier "linear"     0   0.0 "mean" "1e-3"
}

# Submit pipelines for all 4 pretrained model sizes
submit_pipeline "micro"  8
submit_pipeline "small"  16
submit_pipeline "medium" 32
submit_pipeline "large"  64

echo ""
echo "=========================================="
echo "  4 setup + 12 training jobs submitted"
echo "  Training jobs wait for setup to finish"
echo ""
echo "  Monitor: bjobs -g /\$USER/fnirs_perpair_transfer"
echo ""
echo "  After all complete:"
echo "    Compare per-pair results across model sizes"
echo "    Best 20-channel baseline: AUC 0.974 (bn_lstm64)"
echo "=========================================="

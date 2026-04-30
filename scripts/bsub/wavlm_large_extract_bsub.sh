#!/bin/bash
SCRIPT_VERSION="wavlm_large_extract_bsub-v5"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB] span[hosts=1]'
#BSUB -J synchronai-wavlm-large-extract
#BSUB -oo /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/logs/wavlm_large_extract_%J.log

# =============================================================================
# WavLM-large Audio Feature Extraction
#
# Output: data/wavlm_large_features/   shape (49, 1024) per second
#         (1.33× wider than wavlm-base-plus's 768-dim; 24 layers vs 12 internally
#          but blended into one output, so per-file size is ~200 KB vs ~152 KB)
#
# Why: v2 multimodal sweep across architecture/regularization knobs converged
# to val_acc 0.76-0.764 — capacity isn't the bottleneck. Audio expressivity
# (49 timesteps of WavLM-base-plus) is the most likely remaining lever before
# accepting the result. WavLM-large was pretrained on 94k hours of speech
# and is the canonical "more expressive WavLM" choice.
#
# NOT extracting per-layer features: 24 layers × 1024-dim × 59K samples ≈
# 290 GB on disk; defer until blended-large justifies the spend.
#
# Skips env-bootstrap and labels-gen (both already exist). If ml-env or
# data/labels.csv is missing, run scripts/bsub/wavlm_extract_bsub.sh first.
#
# v3: ADDED span[hosts=1] — earlier "audio extraction is exempt from
# fragmentation" framing was wrong for heavy per-item compute. WavLM-large
# is 24 transformer layers, so per-second forward pass is non-trivial CPU
# work that needs intra-process threading; fragmented slots leave most
# cores stranded on remote hosts. The "exempt" rule applies to LIGHT
# extraction (small models, fast features), not transformer extraction.
# =============================================================================

# v2: hardcode SYNCHRONAI_DIR. v1 inherited it from the submitting shell, but
# `bsub < script.sh` only propagates env when LSF_DOCKER_PRESERVE_ENVIRONMENT
# is set in the submit shell — easy to forget. Self-contained scripts are
# fewer footguns. v1 silently failed: empty $SYNCHRONAI_DIR made the labels.csv
# preflight check evaluate "/data/labels.csv" and exit with the wrong error.
export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"

cd "$SYNCHRONAI_DIR"

# Redirect HuggingFace cache to storage (model weights ~1.3 GB)
export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"

# Make synchronai package importable without pip install -e (NFS race risk).
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

# v3: WavLM-large is a 24-layer transformer; per-item compute is non-trivial.
# Without these, PyTorch in docker defaults to 1-2 intra-op threads regardless
# of -n 40, and per-file pace was 15.5s on v2 (10-day projection). With OMP=40
# the same forward pass parallelizes across cores and pace should drop to
# <1s per file. Same lesson as the training-job hangs: extraction is
# parallel ACROSS files but each FILE still needs explicit thread count.
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40

echo "=== [$SCRIPT_VERSION] ==="
echo "SYNCHRONAI_DIR=$SYNCHRONAI_DIR"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_NUM_THREADS=$MKL_NUM_THREADS"

# ml-env/bin/activate doesn't reliably work inside LSF heredocs on this
# cluster; use absolute python path. See docs/ris_bsub_reference.md.
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

# --- Preflight ---
if [ ! -f "$SYNCHRONAI_DIR/data/labels.csv" ]; then
    echo "ERROR: labels.csv not found. Run wavlm_extract_bsub.sh first to generate it."
    exit 1
fi

if [ ! -x "$ML_PY" ]; then
    echo "ERROR: ml-env python not found at $ML_PY"
    exit 1
fi

echo "=== Preflight: ml-env imports ==="
"$ML_PY" -c "import torch, transformers, soundfile; print(f'torch={torch.__version__} transformers={transformers.__version__} soundfile OK')" || {
    echo "ERROR: ml-env missing required packages."
    exit 1
}

# --- Skip if already extracted (only when index covers the full labels set) ---
# v5: tighten the guard. v4 skipped on mere existence of feature_index.csv,
# which let an incomplete previous run (36,517 / 59,250 entries) block a
# resubmit that should have filled the gap. Now we compare entry counts
# and only short-circuit when the existing extraction is complete.
WAVLM_LARGE_DIR="data/wavlm_large_features"
if [ -f "${WAVLM_LARGE_DIR}/feature_index.csv" ]; then
    EXISTING_N=$(tail -n +2 "${WAVLM_LARGE_DIR}/feature_index.csv" | wc -l | tr -d ' ')
    LABELS_N=$(tail -n +2 data/labels.csv | wc -l | tr -d ' ')
    if [ "$EXISTING_N" = "$LABELS_N" ]; then
        echo "=== WavLM-large features already extracted (complete: $EXISTING_N entries) ==="
        echo "  Skipping. Delete ${WAVLM_LARGE_DIR}/feature_index.csv to force re-run."
        exit 0
    else
        echo "=== Resuming WavLM-large extraction ==="
        echo "  Existing index: $EXISTING_N entries (incomplete vs labels.csv: $LABELS_N)"
        echo "  The python script will skip already-extracted .pt files and fill the gap."
        echo "  Old feature_index.csv will be overwritten on completion."
    fi
fi

# --- Extract ---
echo ""
echo "=== Extracting WavLM-large features ==="
echo "  Output: $WAVLM_LARGE_DIR"
echo "  Encoder: wavlm-large (1024-dim, 24 layers, blended)"
echo "  ETA: ~12-18 hours"
echo ""

"$ML_PY" scripts/extract_audio_features.py \
    --labels-file data/labels.csv \
    --output-dir "$WAVLM_LARGE_DIR" \
    --encoder wavlm-large \
    --chunk-duration 1.0 \
    --device auto
extract_rc=$?
if [ $extract_rc -ne 0 ]; then
    echo "ERROR: extraction exited with code $extract_rc — see traceback above."
    echo "  WavLM-large extraction FAILED, no completion banner."
    exit $extract_rc
fi

echo ""
echo "=== WavLM-large extraction complete ==="
echo "  Features: $WAVLM_LARGE_DIR"
echo ""
echo "Next: re-run multimodal sweep with --audio-feature-dir $WAVLM_LARGE_DIR"
echo "  bash scripts/bsub/submit_v2_sweep.sh   # set MM_AUDIO_FEATURE_DIR before"

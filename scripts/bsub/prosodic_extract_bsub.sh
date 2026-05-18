#!/bin/bash
SCRIPT_VERSION="prosodic_extract_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 16000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>16GB] rusage[mem=16GB] span[hosts=1]'
#BSUB -J synchronai-prosodic-extract
#BSUB -oo /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/logs/prosodic_extract_%J.log

# =============================================================================
# eGeMAPS Prosodic Feature Extraction
#
# Output: data/prosodic_features/  shape (~100, 25) per second
#
# Why: D1b + D3 showed WavLM-base-plus features are 95%+ linearly predictable
# from DINOv2 video features — modalities encode redundant scene-level content.
# Prosodic LLDs (pitch, jitter, shimmer, HNR, MFCC 1-4, etc.) target audio
# dimensions mechanistically less likely to overlap with visible behavior.
# If a 25-dim prosodic representation also lands at R²(A→V) > 0.85 with no
# multimodal benefit, the bottleneck is dataset scale and the writeup
# story closes around "data-limited regime, multiple feature types tested."
#
# Resources are light (opensmile is CPU + fast, no model load):
#   -n 4, -M 16GB, span[hosts=1]. ETA ~50min for 59250 entries.
# =============================================================================

# v1: self-contained — hardcode SYNCHRONAI_DIR (per wavlm_large_extract-v2 fix).
export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"

cd "$SYNCHRONAI_DIR"

# Make synchronai package importable without pip install -e (NFS race risk).
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

# Tell PyTorch to use the LSF slot count. (Mostly irrelevant for opensmile,
# which is C++ and threads internally — but our cached audio load does some
# numpy work that benefits.)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=== [$SCRIPT_VERSION] ==="
echo "SYNCHRONAI_DIR=$SYNCHRONAI_DIR"

ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

# --- Preflight ---
if [ ! -f "$SYNCHRONAI_DIR/data/labels.csv" ]; then
    echo "ERROR: labels.csv not found at $SYNCHRONAI_DIR/data/labels.csv"
    exit 1
fi

if [ ! -x "$ML_PY" ]; then
    echo "ERROR: ml-env python not found at $ML_PY"
    exit 1
fi

echo "=== Preflight: ml-env imports ==="
"$ML_PY" -c "import torch, pandas, numpy; print(f'torch={torch.__version__} pandas/numpy OK')" || {
    echo "ERROR: ml-env missing required packages."
    exit 1
}
"$ML_PY" -c "import opensmile; print(f'opensmile={opensmile.__version__} OK')" || {
    echo "ERROR: opensmile not installed in ml-env."
    echo "  Install via:  $ML_PY -m pip install opensmile"
    echo "  Then resubmit this job."
    exit 1
}

# --- Skip if already complete ---
PROSODIC_DIR="data/prosodic_features"
if [ -f "${PROSODIC_DIR}/feature_index.csv" ]; then
    EXISTING_N=$(tail -n +2 "${PROSODIC_DIR}/feature_index.csv" | wc -l | tr -d ' ')
    LABELS_N=$(tail -n +2 data/labels.csv | wc -l | tr -d ' ')
    if [ "$EXISTING_N" = "$LABELS_N" ]; then
        echo "=== Prosodic features already complete ($EXISTING_N entries) ==="
        echo "  Skipping. Delete ${PROSODIC_DIR}/feature_index.csv to force re-run."
        exit 0
    else
        echo "=== Resuming prosodic extraction ==="
        echo "  Existing index: $EXISTING_N entries (incomplete vs labels.csv: $LABELS_N)"
        echo "  Will skip already-extracted .pt files and fill the gap."
    fi
fi

# --- Extract ---
echo ""
echo "=== Extracting eGeMAPS LLDs ==="
echo "  Output: $PROSODIC_DIR"
echo "  Feature set: eGeMAPSv02 LLDs (~100Hz, 25 dims/frame)"
echo "  ETA: ~50min"
echo ""

"$ML_PY" scripts/extract_prosodic_features.py \
    --labels-file data/labels.csv \
    --output-dir "$PROSODIC_DIR" \
    --chunk-duration 1.0
extract_rc=$?
if [ $extract_rc -ne 0 ]; then
    echo "ERROR: extraction exited with code $extract_rc — see traceback above."
    echo "  Prosodic extraction FAILED, no completion banner."
    exit $extract_rc
fi

echo ""
echo "=== Prosodic extraction complete ==="
echo "  Features: $PROSODIC_DIR"
echo ""
echo "Next: re-run multimodal v2 CV with --audio-feature-dir $PROSODIC_DIR"
echo "  MM_AUDIO_FEATURE_DIR=$PROSODIC_DIR \\"
echo "  MM_SAVE_DIR_BASE=runs/multimodal_features/v2_prosodic_cv5 \\"
echo "  MM_NUM_FOLDS=5 \\"
echo "    bash scripts/bsub/submit_kfold.sh"

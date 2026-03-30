#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

# =============================================================================
# WavLM Audio Feature Extraction — Prerequisite for Audio Sweep
#
# Extracts WavLM-base-plus (768-dim) features for all labeled seconds.
# Also ensures ml-env has the transformers package for WavLM.
#
# Must complete BEFORE audio sweep training jobs start (enforced via bsub -w).
# =============================================================================

cd $SYNCHRONAI_DIR

# Redirect HuggingFace cache to storage
export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"

# =============================================================================
# Step 1: Ensure ml-env has transformers (for WavLM)
# =============================================================================

echo "=== Checking ml-env ==="

if [ ! -d "$SYNCHRONAI_DIR/ml-env" ]; then
    echo "Building ml-env from scratch..."
    python -m venv "$SYNCHRONAI_DIR/ml-env"
    source "$SYNCHRONAI_DIR/ml-env/bin/activate"
    pip install --upgrade pip
    pip install --no-cache-dir "numpy>=2.0,<2.5"
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install --no-cache-dir transformers scikit-learn tqdm pandas matplotlib
    pip install --no-cache-dir opencv-python-headless
    pip install --no-cache-dir soundfile imageio-ffmpeg
    pip install -e .
else
    source "$SYNCHRONAI_DIR/ml-env/bin/activate"
    # Ensure transformers + audio deps are installed (may be missing from older ml-env)
    pip install --no-cache-dir transformers soundfile imageio-ffmpeg 2>/dev/null
fi

echo "ml-env ready."
python -c "import torch; import transformers; import soundfile; print('torch + transformers + soundfile OK')"

# Verify ffmpeg is available (needed for extracting audio from video files)
python -c "
from synchronai.data.audio.processing import _check_ffmpeg
assert _check_ffmpeg(), 'ffmpeg not found! Install ffmpeg or imageio-ffmpeg.'
print('ffmpeg OK')
"

# =============================================================================
# Step 2: Preprocess labels (skip if already exists)
# =============================================================================

LABELS_CSV="data/labels.csv"
LABEL_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/synchrony_coding/archive/OLD_synchronycoding_participants/"
VIDEO_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/"

if [ ! -f "${LABELS_CSV}" ]; then
    echo ""
    echo "=== Preprocessing raw data ==="

    python -m synchronai.main --preprocess \
        --label-dir "${LABEL_DIRECTORY}" \
        --video-dir "${VIDEO_DIRECTORY}" \
        --output-csv "${LABELS_CSV}" \
        --conflict-strategy last \
        --label-encoding "a:0,s:1"
else
    echo ""
    echo "=== Labels file already exists: ${LABELS_CSV} ==="
fi

# =============================================================================
# Step 3: Extract WavLM-base-plus features
# =============================================================================

WAVLM_FEATURE_DIR="data/wavlm_baseplus_features"

if [ ! -f "${WAVLM_FEATURE_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting WavLM-base-plus features ==="

    python scripts/extract_audio_features.py \
        --labels-file "${LABELS_CSV}" \
        --output-dir "${WAVLM_FEATURE_DIR}" \
        --encoder wavlm-base-plus \
        --chunk-duration 1.0 \
        --device auto

    echo "WavLM-base-plus extraction complete."
else
    echo ""
    echo "=== WavLM features already extracted ==="
fi

# =============================================================================
# Step 4: Extract per-layer WavLM features (for learnable layer weighting)
# =============================================================================

WAVLM_PERLAYER_DIR="data/wavlm_baseplus_perlayer_features"

if [ ! -f "${WAVLM_PERLAYER_DIR}/feature_index.csv" ]; then
    echo ""
    echo "=== Extracting per-layer WavLM-base-plus features ==="

    python scripts/extract_audio_features.py \
        --labels-file "${LABELS_CSV}" \
        --output-dir "${WAVLM_PERLAYER_DIR}" \
        --encoder wavlm-base-plus \
        --chunk-duration 1.0 \
        --device auto \
        --save-all-layers

    echo "Per-layer extraction complete."
else
    echo ""
    echo "=== Per-layer WavLM features already extracted ==="
fi

echo ""
echo "=== Feature extraction complete ==="
echo "Blended features: ${WAVLM_FEATURE_DIR}"
echo "Per-layer features: ${WAVLM_PERLAYER_DIR}"
echo "ml-env ready for audio sweep training jobs."

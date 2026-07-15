#!/bin/bash
SCRIPT_VERSION="pre_newstudy_features-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 40
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB] span[hosts=1]'
#BSUB -J synchronai-newstudy-features
#BSUB -oo /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/logs/newstudy_features_%J.log

# =============================================================================
# EmoGrow / R56-PCAT DB-DOS feature extraction (video + audio)
#
# Extracts the SAME feature representation the current multimodal model consumes
# so the new data is drop-in compatible:
#   video: DINOv2-base, mean_patch pool, 12fps, 1.0s window, 224px -> 768-dim
#   audio: WavLM-base-plus, 1.0s chunks                              -> 768-dim
#
# Parameterised by two env vars (set before `bsub < ...`, LSF_DOCKER... or
# hardcode). Defaults to EmoGrow.
#   NEWSTUDY_LABELS : labels csv under data/   (labels_emogrow.csv | labels_r56pcat.csv)
#   NEWSTUDY_TAG    : suffix for output dirs   (emogrow | r56pcat)
#
# Outputs:
#   data/dinov2_features_meanpatch_${TAG}/
#   data/wavlm_baseplus_features_${TAG}/
#
# span[hosts=1] + OMP: DINOv2-base and WavLM-base-plus are transformers; per-item
# forward pass is heavy CPU work that needs intra-process threading. Fragmented
# slots strand cores on remote hosts (see CLAUDE.md "LSF -n fragmentation").
#
# NOTE: EmoGrow videos are .mpg (single cam3, low quality). SMOKE-TEST first with
# a small labels subset before committing the full run — see the header echo.
# =============================================================================

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
cd "$SYNCHRONAI_DIR"

NEWSTUDY_LABELS="${NEWSTUDY_LABELS:-labels_emogrow.csv}"
NEWSTUDY_TAG="${NEWSTUDY_TAG:-emogrow}"

export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40

echo "=== [$SCRIPT_VERSION] ==="
echo "NEWSTUDY_LABELS=$NEWSTUDY_LABELS  NEWSTUDY_TAG=$NEWSTUDY_TAG"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_NUM_THREADS=$MKL_NUM_THREADS"

ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"
LABELS_PATH="$SYNCHRONAI_DIR/data/$NEWSTUDY_LABELS"
VIDEO_OUT="data/dinov2_features_meanpatch_${NEWSTUDY_TAG}"
AUDIO_OUT="data/wavlm_baseplus_features_${NEWSTUDY_TAG}"

# --- Preflight ---
if [ ! -x "$ML_PY" ]; then
    echo "ERROR: ml-env python not found at $ML_PY"; exit 1
fi
if [ ! -f "$LABELS_PATH" ]; then
    echo "ERROR: labels file not found: $LABELS_PATH"; exit 1
fi
echo "=== Preflight: ml-env imports ==="
"$ML_PY" -c "import torch, transformers, cv2, soundfile; print('imports OK')" || {
    echo "ERROR: ml-env missing required packages."; exit 1; }
echo "  labels rows: $(tail -n +2 "$LABELS_PATH" | wc -l | tr -d ' ')"

# --- Video: DINOv2-base mean_patch (matches data/dinov2_features_meanpatch) ---
echo ""
echo "=== [1/2] DINOv2-base mean_patch -> $VIDEO_OUT ==="
"$ML_PY" scripts/extract_dinov2_features.py \
    --labels-file "$LABELS_PATH" \
    --output-dir "$VIDEO_OUT" \
    --backbone dinov2-base \
    --pool-mode mean_patch \
    --sample-fps 12.0 \
    --window-seconds 1.0 \
    --frame-size 224 \
    --device auto
rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: DINOv2 extraction exited $rc — see traceback above."; exit $rc
fi

# --- Audio: WavLM-base-plus (matches data/wavlm_baseplus_features) ---
echo ""
echo "=== [2/2] WavLM-base-plus -> $AUDIO_OUT ==="
"$ML_PY" scripts/extract_audio_features.py \
    --labels-file "$LABELS_PATH" \
    --output-dir "$AUDIO_OUT" \
    --encoder wavlm-base-plus \
    --chunk-duration 1.0 \
    --device auto
rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: WavLM extraction exited $rc — see traceback above."; exit $rc
fi

echo ""
echo "=== [$SCRIPT_VERSION] complete for tag=$NEWSTUDY_TAG ==="
echo "  video: $VIDEO_OUT   audio: $AUDIO_OUT"
echo "  video rows: $(tail -n +2 "$VIDEO_OUT/feature_index.csv" 2>/dev/null | wc -l | tr -d ' ')"
echo "  audio rows: $(tail -n +2 "$AUDIO_OUT/feature_index.csv" 2>/dev/null | wc -l | tr -d ' ')"

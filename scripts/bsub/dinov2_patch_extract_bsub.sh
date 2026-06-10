#!/bin/bash
SCRIPT_VERSION="dinov2_patch_extract_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 48000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 8
#BSUB -R 'select[mem>48GB && tmp>20GB] rusage[mem=48GB, tmp=20GB] span[hosts=1]'

# Extract DINOv2 patch-grid features (full (n_frames, 257, 768) token
# sequence per 1-second window). Replaces the broken data/dinov2_features/
# which holds 2D CLS slices but inflated to 9.5 MB each by the
# torch.save view-bloat bug (see feedback_torch_save_view.md and
# extract_dinov2_patch_features.py docstring).
#
# Heavy per-item compute (DINOv2-base forward × 12 frames per second)
# so this is the "needs span+OMP" case from CLAUDE.md.
#
# Wall-clock estimate: 12 frames × 59,250 segments = 711K DINOv2 forwards.
# Dinov2-base on CPU at ~5-15 FPS → 13-40 hours raw inference. Realistic
# budget: 16-30h on 8 cores. Resume support via existing feature_index.csv.

ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

cd "$SYNCHRONAI_DIR"

export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# HF cache on shared storage (home quota is tight)
export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"

LABELS_FILE="${PATCH_LABELS_FILE:-data/labels.csv}"
OUTPUT_DIR="${PATCH_OUTPUT_DIR:-data/dinov2_features_patches}"
BACKBONE="${PATCH_BACKBONE:-dinov2-base}"
SAMPLE_FPS="${PATCH_SAMPLE_FPS:-12}"
WINDOW_SECONDS="${PATCH_WINDOW_SECONDS:-1.0}"
FRAME_SIZE="${PATCH_FRAME_SIZE:-224}"

echo "=== [$SCRIPT_VERSION] ==="
echo "  Labels file:     $LABELS_FILE"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Backbone:        $BACKBONE"
echo "  Sample FPS:      $SAMPLE_FPS (window=${WINDOW_SECONDS}s, frame=${FRAME_SIZE}px)"
echo "  OMP / MKL:       $OMP_NUM_THREADS / $MKL_NUM_THREADS"

# =============================================================================
# Dependency check + install via certifi-based CA bundle (continuumio/anaconda3
# doesn't ship /etc/pki/tls/certs/ca-bundle.crt; pip fails without this).
# =============================================================================
export SSL_CERT_FILE="$("$ML_PY" -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
echo "  CA bundle:       $SSL_CERT_FILE"

echo ""
echo "=== Preflight ==="
"$ML_PY" -c "
import torch, transformers, cv2, numpy, pandas
print(f'torch={torch.__version__} transformers={transformers.__version__} cv2={cv2.__version__}')
" || {
    echo "ERROR: missing import; attempting install"
    "$ML_PY" -m pip install --no-cache-dir --quiet "transformers>=4.40" || {
        echo "ERROR: failed to install transformers"; exit 1
    }
    "$ML_PY" -m pip install --no-cache-dir --quiet --force-reinstall opencv-python-headless || {
        echo "ERROR: failed to install opencv-python-headless"; exit 1
    }
}

if [ ! -f "$LABELS_FILE" ]; then
    echo "ERROR: labels file not found: $LABELS_FILE"
    exit 1
fi

echo ""
echo "=== Starting DINOv2 patch extraction ==="

"$ML_PY" scripts/extract_dinov2_patch_features.py \
    --labels-file "$LABELS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --sample-fps "$SAMPLE_FPS" \
    --window-seconds "$WINDOW_SECONDS" \
    --frame-size "$FRAME_SIZE" \
    --device cpu
extract_rc=$?

if [ $extract_rc -ne 0 ]; then
    echo "ERROR: extraction exited with code $extract_rc"
    exit $extract_rc
fi

echo ""
echo "=== Done ==="
echo "Output:  $OUTPUT_DIR/features/"
echo "Index:   $OUTPUT_DIR/feature_index.csv"

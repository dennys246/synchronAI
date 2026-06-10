#!/bin/bash
SCRIPT_VERSION="pose_extract_bsub-v2"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 48000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 8
#BSUB -R 'select[mem>48GB && tmp>20GB] rusage[mem=48GB, tmp=20GB] span[hosts=1]'

# Probe 3: extract per-frame 2-person pose keypoints for all CARE V0 DB-DOS
# segments. Output goes to data/pose_features/, matching the DINOv2 / WavLM
# feature_index.csv format so the existing multimodal trainer can fuse.
#
# Wall-clock estimate: 12 frames × 59,250 segments = 711K inferences. YOLO26n
# at ~25 FPS CPU = ~8 hours raw inference + I/O overhead. Realistic budget:
# 12-24h on 8 cores.
#
# Heavy per-item compute (YOLO forward pass + video decode), so this DOES
# need span[hosts=1] + OMP_NUM_THREADS per the CLAUDE.md silent-hang note.

ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

cd "$SYNCHRONAI_DIR"

export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

LABELS_FILE="${POSE_LABELS_FILE:-data/labels.csv}"
OUTPUT_DIR="${POSE_OUTPUT_DIR:-data/pose_features}"
MODEL_WEIGHTS="${POSE_MODEL_WEIGHTS:-scripts/yolo26n-pose.pt}"
SAMPLE_FPS="${POSE_SAMPLE_FPS:-12}"
WINDOW_SECONDS="${POSE_WINDOW_SECONDS:-1.0}"
CONF_THRESHOLD="${POSE_CONF_THRESHOLD:-0.25}"
LIMIT="${POSE_LIMIT:-}"

echo "=== [$SCRIPT_VERSION] ==="
echo "  Labels file:     $LABELS_FILE"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Model weights:   $MODEL_WEIGHTS"
echo "  Sample FPS:      $SAMPLE_FPS  (window=${WINDOW_SECONDS}s)"
echo "  Conf threshold:  $CONF_THRESHOLD"
echo "  OMP / MKL:       $OMP_NUM_THREADS / $MKL_NUM_THREADS"
if [ -n "$LIMIT" ]; then
    echo "  LIMIT:           $LIMIT (smoke-test mode)"
fi

# =============================================================================
# Install dependencies into the shared ml-env. ultralytics is the only
# new dep; opencv-python-headless is already present per detect_persons_bsub,
# but force-reinstall the headless variant in case the regular opencv-python
# leaked in. Per CLAUDE.md's "interactive vs non-interactive pip install"
# workaround, this is invoked from a non-interactive bsub job so quota holds.
# =============================================================================
echo ""
echo "=== Ensuring ultralytics + headless opencv in ml-env ==="
# v2: continuumio/anaconda3 doesn't have CA certs at /etc/pki/tls/.
# pip defaults look there, fail with "Could not find a suitable TLS CA
# certificate bundle". The conda env ships its own bundle at
# $CONDA_PREFIX/ssl/cacert.pem (via certifi); point pip at it.
export SSL_CERT_FILE="$("$ML_PY" -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
echo "  Using CA bundle: $SSL_CERT_FILE"
"$ML_PY" -m pip install --no-cache-dir --quiet "ultralytics>=8.3" || {
    echo "ERROR: failed to install ultralytics"
    exit 1
}
"$ML_PY" -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
"$ML_PY" -m pip install --no-cache-dir --quiet --force-reinstall opencv-python-headless || {
    echo "ERROR: failed to install opencv-python-headless"
    exit 1
}

echo ""
echo "=== Preflight ==="
"$ML_PY" -c "
import torch, cv2, ultralytics, numpy
print(f'torch={torch.__version__} cv2={cv2.__version__} ultralytics={ultralytics.__version__} numpy={numpy.__version__}')
" || {
    echo "ERROR: import preflight failed"
    exit 1
}

if [ ! -f "$LABELS_FILE" ]; then
    echo "ERROR: labels file not found: $LABELS_FILE"
    exit 1
fi
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "ERROR: model weights not found: $MODEL_WEIGHTS"
    exit 1
fi

echo ""
echo "=== Starting pose extraction ==="

EXTRA_ARGS=()
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

"$ML_PY" scripts/extract_pose_features.py \
    --labels-file "$LABELS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --model-weights "$MODEL_WEIGHTS" \
    --sample-fps "$SAMPLE_FPS" \
    --window-seconds "$WINDOW_SECONDS" \
    --conf-threshold "$CONF_THRESHOLD" \
    --device cpu \
    --skip-existing \
    "${EXTRA_ARGS[@]}"
extract_rc=$?

if [ $extract_rc -ne 0 ]; then
    echo "ERROR: extraction exited with code $extract_rc"
    exit $extract_rc
fi

echo ""
echo "=== Done ==="
echo "Output:  $OUTPUT_DIR/features/"
echo "Index:   $OUTPUT_DIR/feature_index.csv"

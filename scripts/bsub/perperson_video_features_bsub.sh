#!/bin/bash
SCRIPT_VERSION="perperson_video_features_bsub-v2"
# v2: build script now saves bboxes alongside features + writes
# fallback_per_segment.csv and fallback_per_subject.csv for the
# deep-dive diagnostic. No bsub interface change — same env vars.
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 16000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>16GB && tmp>10GB] rusage[mem=16GB, tmp=10GB] span[hosts=1]'

# Build per-person video features by pooling DINOv2 patch features inside
# pose-derived bboxes. Combines:
#   - data/dinov2_features_patches/  (12, 257, 768) per second
#   - data/pose_features/            (12, 2, 17, 3) per second
# → data/perperson_video_features/   (12, 2, 768) per second
#
# Light per-item compute (numpy patch indexing + mean-pool). Job should
# finish in ~30-60 min for the full ~59K segments. No GPU, no model
# inference — strictly index-and-pool operations on already-extracted
# features.

ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

cd "$SYNCHRONAI_DIR"

export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PATCH_DIR="${PP_PATCH_FEATURE_DIR:-data/dinov2_features_patches}"
POSE_DIR="${PP_POSE_FEATURE_DIR:-data/pose_features}"
OUTPUT_DIR="${PP_OUTPUT_DIR:-data/perperson_video_features}"
MIN_KPT_CONF="${PP_MIN_KPT_CONF:-0.25}"
LIMIT="${PP_LIMIT:-}"

echo "=== [$SCRIPT_VERSION] ==="
echo "  Patch dir:    $PATCH_DIR"
echo "  Pose dir:     $POSE_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Min kpt conf: $MIN_KPT_CONF"
if [ -n "$LIMIT" ]; then
    echo "  LIMIT:        $LIMIT (smoke test)"
fi

# Set CA bundle in case any deps need network.
export SSL_CERT_FILE="$("$ML_PY" -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

echo ""
echo "=== Preflight ==="
"$ML_PY" -c "import torch, numpy; print('imports ok')" || {
    echo "ERROR: missing imports"; exit 1
}

if [ ! -f "$PATCH_DIR/feature_index.csv" ]; then
    echo "ERROR: patch index missing: $PATCH_DIR/feature_index.csv"
    exit 1
fi
if [ ! -f "$POSE_DIR/feature_index.csv" ]; then
    echo "ERROR: pose index missing: $POSE_DIR/feature_index.csv"
    exit 1
fi

echo ""
echo "=== Starting per-person feature construction ==="

EXTRA_ARGS=()
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

"$ML_PY" scripts/build_perperson_video_features.py \
    --patch-feature-dir "$PATCH_DIR" \
    --pose-feature-dir "$POSE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --min-kpt-conf "$MIN_KPT_CONF" \
    --skip-existing \
    "${EXTRA_ARGS[@]}"
rc=$?

if [ $rc -ne 0 ]; then
    echo "ERROR: build exited with code $rc"
    exit $rc
fi

echo ""
echo "=== Done ==="
echo "Output:  $OUTPUT_DIR/features/"
echo "Index:   $OUTPUT_DIR/feature_index.csv"

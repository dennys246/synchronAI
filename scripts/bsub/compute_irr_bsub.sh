#!/bin/bash
SCRIPT_VERSION="compute_irr_bsub-v1"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 8000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>8GB] rusage[mem=8GB] span[hosts=1]'

# Compute inter-rater reliability (Cohen's kappa, per-session agreement,
# difficulty scores) for the CARE synchrony annotations. Output drives
# the "what's the architectural ceiling we can possibly hit" decision —
# the implied AUC ceiling from kappa tells us whether our 0.72 CV mean
# is close to the noise floor or whether there's real architectural
# headroom left.
#
# Reads multi-annotator xlsx files from $IRR_LABEL_DIR with structure:
#     {subject_id}/{session}/*.xlsx
# Sessions with 2+ xlsx files are the multi-annotator overlap set.
#
# Output:  $IRR_OUTPUT_DIR/irr_report.json + irr_summary.txt + plots

ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

cd "$SYNCHRONAI_DIR"

export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

LABEL_DIR="${IRR_LABEL_DIR:?IRR_LABEL_DIR must be set (e.g. /storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/label_data/)}"
OUTPUT_DIR="${IRR_OUTPUT_DIR:-runs/irr_analysis}"
LABELS_CSV="${IRR_LABELS_CSV:-data/labels.csv}"
ENCODING="${IRR_ENCODING:-a:0,s:1}"

echo "=== [$SCRIPT_VERSION] ==="
echo "  Label dir:    $LABEL_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Labels CSV:   $LABELS_CSV"
echo "  Encoding:     $ENCODING"

# Set up CA bundle in case any deps need network (compute_irr.py is local
# but doesn't hurt to have).
export SSL_CERT_FILE="$("$ML_PY" -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

echo ""
echo "=== Preflight ==="
"$ML_PY" -c "import pandas, openpyxl, matplotlib, sklearn; print('imports ok')" || {
    echo "ERROR: missing import; attempting install"
    "$ML_PY" -m pip install --no-cache-dir --quiet openpyxl scikit-learn matplotlib || {
        echo "ERROR: failed to install deps"; exit 1
    }
}

if [ ! -d "$LABEL_DIR" ]; then
    echo "ERROR: label dir not found: $LABEL_DIR"
    exit 1
fi

EXTRA_ARGS=()
if [ -n "$LABELS_CSV" ] && [ -f "$LABELS_CSV" ]; then
    EXTRA_ARGS+=(--labels-csv "$LABELS_CSV")
fi

echo ""
echo "=== Running compute_irr.py ==="

"$ML_PY" scripts/compute_irr.py \
    --label-dir "$LABEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --encoding "$ENCODING" \
    "${EXTRA_ARGS[@]}"
rc=$?

if [ $rc -ne 0 ]; then
    echo "ERROR: compute_irr.py exited with code $rc"
    exit $rc
fi

echo ""
echo "=== Done ==="
echo "Summary:  $OUTPUT_DIR/irr_summary.txt"
echo "Full JSON: $OUTPUT_DIR/irr_report.json"
echo ""
echo "Inspect the headline number first:"
echo "  cat $OUTPUT_DIR/irr_summary.txt | head -40"

#!/bin/bash
SCRIPT_VERSION="pre_fnirs_compute_qc_bsub-v1"
# =============================================================================
# fNIRS QC Tier Pre-Computation
#
# Runs QC (SCI, cardiac, SNR) on all recordings and writes qc_tiers.csv.
# Run this ONCE before extraction — extraction uses --qc-cache to skip
# inline QC (saves ~50% of extraction time).
# =============================================================================

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

DATE=$(date +'%m-%d')
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

QC_OUTPUT="$SYNCHRONAI_DIR/data/qc_tiers.csv"

echo "=== [$SCRIPT_VERSION] ==="

bsub -J "synchronai-qc-tiers-$DATE" \
     -G compute-perlmansusan \
     -q general \
     -m general \
     -M 16000000 \
     -a 'docker(continuumio/anaconda3)' \
     -n 8 \
     -R 'select[mem>16GB] rusage[mem=16GB]' \
     -oo "$LOG_DIR/fnirs_compute_qc_$DATE.log" \
     -g /$USER/fnirs_extract \
     << QC_EOF
echo "=== [$SCRIPT_VERSION] ==="
cd $SYNCHRONAI_DIR
. "$SYNCHRONAI_DIR/ml-env/bin/activate"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"

python scripts/compute_fnirs_qc.py \
    --data-dirs "$FNIRS_DIRS" \
    --output "$QC_OUTPUT" \
    --sci-threshold 0.40 \
    --snr-threshold 2.0 \
    --cardiac-peak-ratio 2.0 \
    --no-require-cardiac

echo "=== QC complete ==="
QC_EOF

echo "  Monitor: bjobs -g /\$USER/fnirs_extract"
echo "  Output: $QC_OUTPUT"
echo ""
echo "  After QC completes, run extraction:"
echo "    sh scripts/bsub/pre_fnirs_extract_features_bsub.sh"

#!/bin/bash

set -euo pipefail

NUMBA_DISABLE_JIT=1
MNE_USE_NUMBA=0

DEFAULT_EPOCHS=0
DEFAULT_LR=1e-4
DEFAULT_BATCH_SIZE=8
DEFAULT_SEGMENTS_PER_RECORDING=4
DEFAULT_UNET_BASE_WIDTH=64
DEFAULT_UNET_DEPTH=3
DEFAULT_UNET_TIME_EMBED_DIM=128
DEFAULT_MAX_RECORDINGS=0
DEFAULT_RECORDINGS_PER_BATCH=4
DEFAULT_DISABLE_NUMBA=0
DEFAULT_DURATION_SECONDS=120.0

EPOCHS="$DEFAULT_EPOCHS"
LR="$DEFAULT_LR"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
SEGMENTS_PER_RECORDING="$DEFAULT_SEGMENTS_PER_RECORDING"
UNET_BASE_WIDTH="$DEFAULT_UNET_BASE_WIDTH"
UNET_DEPTH="$DEFAULT_UNET_DEPTH"
UNET_TIME_EMBED_DIM="$DEFAULT_UNET_TIME_EMBED_DIM"
MAX_RECORDINGS="$DEFAULT_MAX_RECORDINGS"
RECORDINGS_PER_BATCH="$DEFAULT_RECORDINGS_PER_BATCH"
VERBOSEE=0
DISABLE_NUMBA="$DEFAULT_DISABLE_NUMBA"
DURATION_SECONDS="$DEFAULT_DURATION_SECONDS"
SAVE_ROOT="${SYNCHRONAI_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}/runs/fnirs_diffusion"

usage() {
  cat <<'EOF'
Usage: scripts/generative_pretrain.sh [--epochs N] [--lr LR] [--max-recordings N] [--recordings-per-batch N] [--duration-seconds S] [--verbosee] [--disable-numba]

Trains a single unified fNIRS diffusion model on all CARE and P_CAT datasets.
Defaults: --epochs 0 (unlimited) --lr 1e-4 --max-recordings 0 (unlimited) --recordings-per-batch 4 (load 4 at once) --duration-seconds 120
Note: epochs=0 means train forever (use Ctrl+C to stop), or specify --epochs N for a fixed number
EOF
}

require_arg() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "Missing value for $flag" >&2
    usage >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      require_arg "$1" "${2:-}"
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      require_arg "$1" "${2:-}"
      LR="$2"
      shift 2
      ;;
    --max-recordings)
      require_arg "$1" "${2:-}"
      MAX_RECORDINGS="$2"
      shift 2
      ;;
    --recordings-per-batch)
      require_arg "$1" "${2:-}"
      RECORDINGS_PER_BATCH="$2"
      shift 2
      ;;
    --verbosee)
      VERBOSEE=1
      shift 1
      ;;
    --disable-numba)
      DISABLE_NUMBA=1
      shift 1
      ;;
    --duration-seconds)
      require_arg "$1" "${2:-}"
      DURATION_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# All dataset paths - treated uniformly for unified training
fNIRS_DATASETS=(
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T1/nirs_data/dbdos/"
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T3/nirs_data/dbdos/"
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/PSU_data/T5/nirs_data/dbdos/"
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T1/nirs_data/dbdos/"
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T3/nirs_data/dbdos/"
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R01/data/PSU_share/WUSTL_data/T5/nirs_data/dbdos/"
  "/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/NIRS_data/"
)

# Combine all dataset paths into a single colon-separated string
ALL_DATA_DIRS=""
for dataset_dir in "${fNIRS_DATASETS[@]}"; do
  if [[ -e "$dataset_dir" ]]; then
    if [[ -z "$ALL_DATA_DIRS" ]]; then
      ALL_DATA_DIRS="$dataset_dir"
    else
      ALL_DATA_DIRS="${ALL_DATA_DIRS}:${dataset_dir}"
    fi
  else
    echo "Warning: Skipping missing dataset dir: $dataset_dir" >&2
  fi
done

if [[ -z "$ALL_DATA_DIRS" ]]; then
  echo "Error: No valid dataset directories found" >&2
  exit 1
fi

echo "=== Training unified fNIRS diffusion model on CARE + P_CAT datasets ==="
echo "Dataset directories: $ALL_DATA_DIRS"
echo ""

CMD=(
  python -m synchronai.main
  --fnirs
  --train diffusion
  --trace
  --data-dir "$ALL_DATA_DIRS"
  --save-dir "$SAVE_ROOT"
  --batch-size "$BATCH_SIZE"
  --segments-per-recording "$SEGMENTS_PER_RECORDING"
  --unet-base-width "$UNET_BASE_WIDTH"
  --unet-depth "$UNET_DEPTH"
  --unet-time-embed-dim "$UNET_TIME_EMBED_DIM"
  --epochs "$EPOCHS"
  --learning-rate "$LR"
  --max-recordings "$MAX_RECORDINGS"
  --recordings-per-batch "$RECORDINGS_PER_BATCH"
  --duration-seconds "$DURATION_SECONDS"
)
if [[ "$VERBOSEE" -eq 1 ]]; then
  CMD+=(--verbosee)
fi
if [[ "$DISABLE_NUMBA" -eq 1 ]]; then
  CMD+=(--disable-numba)
fi
"${CMD[@]}"

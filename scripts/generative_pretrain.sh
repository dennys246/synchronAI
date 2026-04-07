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
DEFAULT_UNET_DROPOUT=0.15
DEFAULT_MAX_RECORDINGS=0
DEFAULT_RECORDINGS_PER_BATCH=4
DEFAULT_DISABLE_NUMBA=0
DEFAULT_DURATION_SECONDS=60.0
DEFAULT_VAL_FRACTION=0.2
DEFAULT_LR_SCHEDULE=cosine_restarts
DEFAULT_EVAL_GEN_EVERY=10
DEFAULT_SCI_THRESHOLD=0.5
DEFAULT_SNR_THRESHOLD=5.0
DEFAULT_CARDIAC_PEAK_RATIO=2.0

EPOCHS="$DEFAULT_EPOCHS"
LR="$DEFAULT_LR"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
SEGMENTS_PER_RECORDING="$DEFAULT_SEGMENTS_PER_RECORDING"
UNET_BASE_WIDTH="$DEFAULT_UNET_BASE_WIDTH"
UNET_DEPTH="$DEFAULT_UNET_DEPTH"
UNET_TIME_EMBED_DIM="$DEFAULT_UNET_TIME_EMBED_DIM"
UNET_DROPOUT="$DEFAULT_UNET_DROPOUT"
MAX_RECORDINGS="$DEFAULT_MAX_RECORDINGS"
RECORDINGS_PER_BATCH="$DEFAULT_RECORDINGS_PER_BATCH"
VERBOSEE=0
DISABLE_NUMBA="$DEFAULT_DISABLE_NUMBA"
DURATION_SECONDS="$DEFAULT_DURATION_SECONDS"
VAL_FRACTION="$DEFAULT_VAL_FRACTION"
LR_SCHEDULE="$DEFAULT_LR_SCHEDULE"
EVAL_GEN_EVERY="$DEFAULT_EVAL_GEN_EVERY"
PER_PAIR=0
ENABLE_QC=0
SCI_THRESHOLD="$DEFAULT_SCI_THRESHOLD"
SNR_THRESHOLD="$DEFAULT_SNR_THRESHOLD"
CARDIAC_PEAK_RATIO="$DEFAULT_CARDIAC_PEAK_RATIO"
REQUIRE_CARDIAC=1
PEAK_POWER_LOW=""
PEAK_POWER_HIGH=""
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
    --lr-schedule)
      require_arg "$1" "${2:-}"
      LR_SCHEDULE="$2"
      shift 2
      ;;
    --save-dir)
      require_arg "$1" "${2:-}"
      SAVE_ROOT="$2"
      shift 2
      ;;
    --unet-base-width)
      require_arg "$1" "${2:-}"
      UNET_BASE_WIDTH="$2"
      shift 2
      ;;
    --unet-depth)
      require_arg "$1" "${2:-}"
      UNET_DEPTH="$2"
      shift 2
      ;;
    --eval-gen-every)
      require_arg "$1" "${2:-}"
      EVAL_GEN_EVERY="$2"
      shift 2
      ;;
    --per-pair)
      PER_PAIR=1
      shift 1
      ;;
    --enable-qc)
      ENABLE_QC=1
      shift 1
      ;;
    --sci-threshold)
      require_arg "$1" "${2:-}"
      SCI_THRESHOLD="$2"
      shift 2
      ;;
    --snr-threshold)
      require_arg "$1" "${2:-}"
      SNR_THRESHOLD="$2"
      shift 2
      ;;
    --cardiac-peak-ratio)
      require_arg "$1" "${2:-}"
      CARDIAC_PEAK_RATIO="$2"
      shift 2
      ;;
    --no-require-cardiac)
      REQUIRE_CARDIAC=0
      shift 1
      ;;
    --peak-power-low)
      require_arg "$1" "${2:-}"
      PEAK_POWER_LOW="$2"
      shift 2
      ;;
    --peak-power-high)
      require_arg "$1" "${2:-}"
      PEAK_POWER_HIGH="$2"
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
  --unet-dropout "$UNET_DROPOUT"
  --epochs "$EPOCHS"
  --learning-rate "$LR"
  --max-recordings "$MAX_RECORDINGS"
  --recordings-per-batch "$RECORDINGS_PER_BATCH"
  --duration-seconds "$DURATION_SECONDS"
  --val-fraction "$VAL_FRACTION"
  --lr-schedule "$LR_SCHEDULE"
  --eval-gen-every "$EVAL_GEN_EVERY"
)
if [[ "$PER_PAIR" -eq 1 ]]; then
  CMD+=(--per-pair)
fi
if [[ "$ENABLE_QC" -eq 1 ]]; then
  CMD+=(--enable-qc)
  CMD+=(--sci-threshold "$SCI_THRESHOLD")
  CMD+=(--snr-threshold "$SNR_THRESHOLD")
  CMD+=(--cardiac-peak-ratio "$CARDIAC_PEAK_RATIO")
  if [[ "$REQUIRE_CARDIAC" -eq 0 ]]; then
    CMD+=(--no-require-cardiac)
  fi
  if [[ -n "$PEAK_POWER_LOW" ]]; then
    CMD+=(--peak-power-low "$PEAK_POWER_LOW")
  fi
  if [[ -n "$PEAK_POWER_HIGH" ]]; then
    CMD+=(--peak-power-high "$PEAK_POWER_HIGH")
  fi
fi
if [[ "$VERBOSEE" -eq 1 ]]; then
  CMD+=(--verbosee)
fi
if [[ "$DISABLE_NUMBA" -eq 1 ]]; then
  CMD+=(--disable-numba)
fi
"${CMD[@]}"

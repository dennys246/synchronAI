#!/bin/bash

set -euo pipefail

DEFAULT_N_SAMPLES=3
DEFAULT_SEED=42
SAVE_ROOT="runs/fnirs_diffusion"

N_SAMPLES="$DEFAULT_N_SAMPLES"
SEED="$DEFAULT_SEED"
VERBOSEE=0

usage() {
  cat <<'EOF'
Usage: scripts/generate_fnirs.sh [--n-samples N] [--seed SEED] [--verbosee]

Generates synthetic fNIRS hemoglobin samples using the trained unified diffusion model.
Saves generated samples and plots to runs/fnirs_diffusion/generated/

Defaults: --n-samples 3 --seed 42
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
    --n-samples)
      require_arg "$1" "${2:-}"
      N_SAMPLES="$2"
      shift 2
      ;;
    --seed)
      require_arg "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --verbosee)
      VERBOSEE=1
      shift 1
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

if [[ ! -d "$SAVE_ROOT" ]]; then
  echo "Error: Model directory not found: $SAVE_ROOT" >&2
  echo "Please train the model first using scripts/generative_pretrain.sh" >&2
  exit 1
fi

if [[ ! -f "$SAVE_ROOT/fnirs_diffusion_config.json" ]]; then
  echo "Error: Config file not found: $SAVE_ROOT/fnirs_diffusion_config.json" >&2
  echo "Please train the model first using scripts/generative_pretrain.sh" >&2
  exit 1
fi

echo "=== Generating fNIRS synthetic samples ==="
echo "Model directory: $SAVE_ROOT"
echo "Number of samples: $N_SAMPLES"
echo "Random seed: $SEED"
echo ""

CMD=(
  python -m synchronai.main
  --fnirs
  --generate diffusion
  --trace
  --save-dir "$SAVE_ROOT"
  --n-samples "$N_SAMPLES"
  --seed "$SEED"
)

if [[ "$VERBOSEE" -eq 1 ]]; then
  CMD+=(--verbosee)
fi

"${CMD[@]}"

echo ""
echo "=== Generation complete ==="
echo "Generated samples saved to: $SAVE_ROOT/generated/"

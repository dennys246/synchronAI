#!/bin/bash
SCRIPT_VERSION="submit_v2_sweep-v2"
# =============================================================================
# Submit a multimodal v2 sweep batch in parallel. Each variant inherits v6+
# BSub defaults (LR=5e-5, batch=128, epochs=30, warmup=5, patience=10,
# early-stop on val_loss, span[hosts=1], OMP=4) and overrides only the knobs
# being tested.
#
# Variants:
#   baseline_v6     — re-run with val_loss stopping + multi-criterion ckpts
#   higher_capacity — hidden=128 (test underfitting)
#   lower_capacity  — hidden=32  (test subject memorization)
#   more_reg        — dropout=0.5, wd=3e-2 (push past epoch-5 overfit cliff)
#
# Save dirs: runs/multimodal_features/${MM_SWEEP_TAG}<variant>/
#   - Default tag "v2_" → preserves the original v1 naming
#     (v2_baseline_v6, v2_higher_capacity, ...)
#   - Override to keep parallel sweeps non-colliding, e.g.:
#       MM_SWEEP_TAG=v2_wavlm_large_  → v2_wavlm_large_baseline_v6/, etc.
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"
SWEEP_TAG="${MM_SWEEP_TAG:-v2_}"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

echo "=== [$SCRIPT_VERSION] Submitting v2 sweep (4 variants, tag=${SWEEP_TAG}) ==="
if [ -n "${MM_AUDIO_FEATURE_DIR:-}" ]; then
    echo "    audio features: $MM_AUDIO_FEATURE_DIR"
fi
if [ -n "${MM_VIDEO_FEATURE_DIR:-}" ]; then
    echo "    video features: $MM_VIDEO_FEATURE_DIR"
fi

submit_variant() {
    local variant="$1"; shift
    local save_dir="runs/multimodal_features/${SWEEP_TAG}${variant}"
    echo ""
    echo "--- Submitting ${SWEEP_TAG}${variant} → $save_dir ---"
    # Subshell so env-var settings don't leak between submissions. Invoke the
    # launcher via `sh` (matching its #!/bin/sh shebang) so permission bits
    # don't matter — NFS/GPFS sometimes drops the executable bit across mounts.
    (
        export MM_SAVE_DIR="$save_dir"
        # shellcheck disable=SC2068
        for kv in $@; do
            export "$kv"
        done
        sh "$LAUNCHER"
    )
}

submit_variant "baseline_v6"

submit_variant "higher_capacity" \
    "MM_VIDEO_HIDDEN=128" \
    "MM_AUDIO_HIDDEN=128" \
    "MM_HEAD_HIDDEN=128"

submit_variant "lower_capacity" \
    "MM_VIDEO_HIDDEN=32" \
    "MM_AUDIO_HIDDEN=32" \
    "MM_HEAD_HIDDEN=32"

submit_variant "more_reg" \
    "MM_DROPOUT=0.5" \
    "MM_WEIGHT_DECAY=3e-2"

echo ""
echo "=== All 4 variants submitted. Check status with: bjobs ==="
echo "Logs land in: scripts/bsub/logs/synchronai_multimodal_features_<DATE>_<JOBID>.log"

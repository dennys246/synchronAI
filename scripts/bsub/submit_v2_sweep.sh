#!/bin/bash
SCRIPT_VERSION="submit_v2_sweep-v1"
# =============================================================================
# Submit the first batch of multimodal v2 sweep variants in parallel.
# Each variant inherits v6 BSub defaults (LR=5e-5, batch=128, epochs=30,
# warmup=5, patience=10, early-stop on val_loss, span[hosts=1], OMP=8) and
# overrides only the knobs being tested.
#
# Variants:
#   v2_baseline_v6     — re-run with val_loss stopping + multi-criterion ckpts
#   v2_higher_capacity — hidden=128 (test underfitting)
#   v2_lower_capacity  — hidden=32  (test subject memorization)
#   v2_more_reg        — dropout=0.5, wd=3e-2 (push past epoch-5 overfit cliff)
#
# Each writes to its own runs/multimodal_features/<name>/ and gets its own LSF
# job ID + log file via -oo "..._%J.log".
# =============================================================================

LAUNCHER="$(dirname "$0")/pre_multimodal_from_features_bsub.sh"

if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher not found: $LAUNCHER"
    exit 1
fi

echo "=== [$SCRIPT_VERSION] Submitting v2 sweep batch 1 (4 variants) ==="

submit_variant() {
    local name="$1"; shift
    echo ""
    echo "--- Submitting $name ---"
    # Subshell so env-var settings don't leak between submissions. Invoke the
    # launcher via `sh` (matching its #!/bin/sh shebang) so permission bits
    # don't matter — NFS/GPFS sometimes drops the executable bit across mounts.
    (
        export MM_SAVE_DIR="runs/multimodal_features/$name"
        # shellcheck disable=SC2068
        for kv in $@; do
            export "$kv"
        done
        sh "$LAUNCHER"
    )
}

submit_variant "v2_baseline_v6"

submit_variant "v2_higher_capacity" \
    "MM_VIDEO_HIDDEN=128" \
    "MM_AUDIO_HIDDEN=128" \
    "MM_HEAD_HIDDEN=128"

submit_variant "v2_lower_capacity" \
    "MM_VIDEO_HIDDEN=32" \
    "MM_AUDIO_HIDDEN=32" \
    "MM_HEAD_HIDDEN=32"

submit_variant "v2_more_reg" \
    "MM_DROPOUT=0.5" \
    "MM_WEIGHT_DECAY=3e-2"

echo ""
echo "=== All 4 variants submitted. Check status with: bjobs ==="
echo "Logs land in: scripts/bsub/logs/synchronai_multimodal_features_<DATE>_<JOBID>.log"

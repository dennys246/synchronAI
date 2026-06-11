#!/bin/bash
SCRIPT_VERSION="run_fnirs_attribution_tile-v1"
# =============================================================================
# fNIRS Attribution pipeline — TILE re-run launcher (Option B).
#
# The three attribution bsubs (band_attribution, window_position,
# signal_diagnostics) are fully env-overridable, so this just points them at the
# _tile artifacts and fresh _tile output dirs — no copies needed, and the
# existing attribution results stay intact until swap.
#
# Analyzes the {SIZE}_lstm64 classifier (paper used small_lstm64). Override the
# size with ATTR_SIZE=large.
#
# Run AFTER: the tile classifier sweep produced
#   runs/fnirs_child_adult_sweep_tile/{SIZE}_lstm64/best.pt
# =============================================================================

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

SIZE="${ATTR_SIZE:-small}"
CLF="${SIZE}_lstm64"
BSUB_DIR="$SYNCHRONAI_DIR/scripts/bsub"

# Shared _tile artifact paths (exported so the child bsub scripts pick them up
# via their ${VAR:-default} and bake them into the jobs).
export ENCODER_PT="$SYNCHRONAI_DIR/runs/fnirs_perpair_${SIZE}_tile/fnirs_unet_encoder.pt"
export CLASSIFIER_PT="$SYNCHRONAI_DIR/runs/fnirs_child_adult_sweep_tile/${CLF}/best.pt"
export FEATURE_DIR="$SYNCHRONAI_DIR/data/fnirs_perpair_${SIZE}_tile_features"
export OUT_DIR="$SYNCHRONAI_DIR/results/band_attributions_val_tile"

echo "=== [$SCRIPT_VERSION] attribution (tile), classifier=$CLF ==="
echo "  encoder    : $ENCODER_PT"
echo "  classifier : $CLASSIFIER_PT"
echo "  features   : $FEATURE_DIR"
echo "  out        : $OUT_DIR"

for f in "$CLASSIFIER_PT" "$ENCODER_PT" "$FEATURE_DIR/feature_index.csv"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: required input missing: $f (has the tile sweep + extraction finished?)"
        exit 1
    fi
done

# 1) Band attribution (Stage 1 IG + Stage 2 group stats, both internal).
echo "--- submitting band attribution ---"
bash "$BSUB_DIR/fnirs_band_attribution_bsub.sh"

# 2) Window-position analysis (independent of band attribution).
export ORIGINAL_CLASSIFIER="$CLASSIFIER_PT"
export ORIG_EVAL_DIR="$SYNCHRONAI_DIR/runs/fnirs_child_adult_sweep_tile/${CLF}/window_position_analysis"
export WIN0_SAVE_DIR="$SYNCHRONAI_DIR/runs/fnirs_child_adult_sweep_tile/${CLF}_window0"
export WIN0_EVAL_DIR="$WIN0_SAVE_DIR/window_position_analysis"
echo "--- submitting window-position analysis ---"
bash "$BSUB_DIR/fnirs_window_position_bsub.sh"

# 3) Signal diagnostics — reads band-attribution output, so run it AFTER band
#    attribution completes. Submit manually once OUT_DIR is populated:
echo ""
echo "=== band attribution + window-position submitted ==="
echo "After band attribution finishes (results in $OUT_DIR), run diagnostics:"
echo "  INPUT_DIR=$OUT_DIR ENCODER_PT=$ENCODER_PT \\"
echo "    bash $BSUB_DIR/fnirs_signal_diagnostics_bsub.sh"
echo "Monitor: bjobs"

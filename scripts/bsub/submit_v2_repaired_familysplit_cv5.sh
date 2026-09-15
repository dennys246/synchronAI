#!/bin/bash
# Re-establish the v2 multimodal baseline on REPAIRED CARE labels.
#
# The previous baseline (runs/multimodal_features/arch_sweep_familysplit/
# arch_v2_orig_fold*) was trained on data/labels.csv, where 85.7% of rows carried
# timestamps past the end of the video (MM:SS decoded as HH:MM:SS -> 60x). Frame
# reads past end-of-video silently substituted a BLACK FRAME, so those windows
# became constant features. See docs/results/
# multimodal_label_corruption_and_reextraction.md.
#
# This run changes ONE thing: the feature dirs, re-extracted from
# data/labels_care_repaired.csv (ffprobe-validated, 65,547 rows, 100% in range).
# Every hyperparameter is left at the multimodal_from_features_bsub.sh default,
# which already equals the arch_v2_orig config (v2, 30 epochs, batch 128,
# lr 5e-5, wd 1e-2, dropout 0.3, hidden 64/64/64, warmup 5, patience 10,
# early-stop val_loss, group_key family_id). Do not add flags here — a
# difference other than the data makes the comparison uninterpretable.
#
# No fallback filter: it existed to DROP the out-of-range rows. The repair fixes
# them instead, so filtering now would discard valid data.
#
# LAUNCHER (run directly, NOT via `bsub <`). Dry-run first:
#   bash scripts/bsub/submit_v2_repaired_familysplit_cv5.sh            # prints only
#   DRY_RUN=0 bash scripts/bsub/submit_v2_repaired_familysplit_cv5.sh  # submits
SCRIPT_VERSION="submit_v2_repaired_familysplit_cv5-v1"
echo "=== [$SCRIPT_VERSION] ==="

set -u
SYNCHRONAI_DIR="${SYNCHRONAI_DIR:-/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI}"
cd "$SYNCHRONAI_DIR" || { echo "ERROR: cannot cd $SYNCHRONAI_DIR"; exit 1; }

TRAIN_BSUB="scripts/bsub/multimodal_from_features_bsub.sh"
LOGDIR="scripts/bsub/logs"
OUTROOT="runs/multimodal_features"
SAVESUB="repaired_familysplit/v2_baseline"
DRY_RUN="${DRY_RUN:-1}"
NUM_FOLDS="${NUM_FOLDS:-5}"
mkdir -p "$LOGDIR"

VDIR="data/dinov2_features_meanpatch_care_repaired"
ADIR="data/wavlm_baseplus_features_care_repaired"

# 65,547 rows x (audio 49x768 + video 12x768) float32 ~= 12.2GB resident.
# 24GB was enough for the 59,250-row runs; 32GB covers the larger set without
# the longer pend queue that the wrapper's own 48GB reservation draws.
MEM_GB=32

for d in "$VDIR" "$ADIR"; do
  if [ ! -f "$d/feature_index.csv" ]; then
    echo "ERROR: missing $d/feature_index.csv — run the re-extraction first."; exit 1
  fi
done
echo "  video rows: $(tail -n +2 "$VDIR/feature_index.csv" | wc -l | tr -d ' ')"
echo "  audio rows: $(tail -n +2 "$ADIR/feature_index.csv" | wc -l | tr -d ' ')"

n=0
for ((f=0; f<NUM_FOLDS; f++)); do
  SAVE="$OUTROOT/${SAVESUB}_fold${f}"
  if [ "${FORCE:-0}" != "1" ] && [ -f "$SAVE/history.json" ]; then
    echo "skip fold$f (history.json exists)"; continue
  fi
  n=$((n+1))
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry] fold$f -> $SAVE"
  else
    env MM_ARCH=v2 MM_VIDEO_FEATURE_DIR="$VDIR" MM_AUDIO_FEATURE_DIR="$ADIR" \
        MM_SAVE_DIR="$SAVE" MM_GROUP_KEY=family_id \
        MM_NUM_FOLDS="$NUM_FOLDS" MM_FOLD_IDX="$f" \
        bsub -M "${MEM_GB}000000" \
             -R "select[mem>${MEM_GB}GB] rusage[mem=${MEM_GB}GB] span[hosts=1]" \
             -oo "$LOGDIR/v2repaired_fold${f}_%J.log" < "$TRAIN_BSUB"
  fi
done

echo "=== $n jobs $([ "$DRY_RUN" = 1 ] && echo 'would be submitted (DRY-RUN — set DRY_RUN=0 to submit)' || echo submitted) ==="
echo "Summarize (prints pooled AND within-recording AUC):"
echo "  python scripts/summarize_hp_sweep.py $OUTROOT/repaired_familysplit"
echo "Compare against the corrupted-label baseline:"
echo "  $OUTROOT/arch_sweep_familysplit/arch_v2_orig_fold*"

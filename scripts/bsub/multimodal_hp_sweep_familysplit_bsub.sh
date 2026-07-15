#!/bin/bash
# Launcher: family_id 5-fold CV hyperparameter + arch-sanity sweep on the
# ORIGINAL 49 (CARE + P-CAT, data/dinov2_features_meanpatch + wavlm_baseplus_features).
#
# WHY: the subject_id split leaked ~0.10 AUC via a shared-parent confound (two
# 5-digit children of one family straddling train/val). All prior architecture
# and hyperparameter selection was done on that leaky metric, so we re-select on
# the leak-free family_id metric — with k-fold CV, not a single split, so the
# choice isn't made on one lucky/high-variance val draw. Targets the observed
# symptom: honest runs overfit and peak at ~epoch 1, so the grid leans on LR /
# schedule / regularization / capacity. See feedback_multimodal_family_split_leak.
#
# This is a LAUNCHER (run directly, NOT via `bsub <`). It submits one job per
# (config, fold) by reusing scripts/bsub/multimodal_from_features_bsub.sh (v19)
# with per-job MM_* env. Each inner job echoes its own version.
#
# Usage (dry-run FIRST per cluster discipline, then submit):
#   bash scripts/bsub/multimodal_hp_sweep_familysplit_bsub.sh          # prints commands only
#   DRY_RUN=0 bash scripts/bsub/multimodal_hp_sweep_familysplit_bsub.sh # actually submits
SCRIPT_VERSION="multimodal_hp_sweep_familysplit-v1"
echo "=== [$SCRIPT_VERSION] ==="

set -u
SYNCHRONAI_DIR="${SYNCHRONAI_DIR:-/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI}"
cd "$SYNCHRONAI_DIR" || { echo "ERROR: cannot cd $SYNCHRONAI_DIR"; exit 1; }

TRAIN_BSUB="scripts/bsub/multimodal_from_features_bsub.sh"
LOGDIR="scripts/bsub/logs"
OUTROOT="runs/multimodal_features/hpsweep_familysplit"
DRY_RUN="${DRY_RUN:-1}"                 # default: print only. DRY_RUN=0 to submit.
NUM_FOLDS="${NUM_FOLDS:-5}"
# Original-49 set is ~11 GB preloaded → 24 GB is ample and kinder than the 48 GB
# the merged run needs. Keep span[hosts=1] (CPU-hang fix). -M / -R on the bsub
# command line override the training script's #BSUB directives for this sweep.
MEM_MB=24000000
RES='select[mem>24GB] rusage[mem=24GB] span[hosts=1]'

VIDEO_DIR="data/dinov2_features_meanpatch"
AUDIO_DIR="data/wavlm_baseplus_features"

if [ ! -f "$VIDEO_DIR/feature_index.csv" ] || [ ! -f "$AUDIO_DIR/feature_index.csv" ]; then
    echo "ERROR: original-49 feature dirs not found ($VIDEO_DIR / $AUDIO_DIR)"; exit 1
fi
mkdir -p "$LOGDIR"

# config: name | ARCH | LR | WARMUP | WD | DROPOUT | HIDDEN | EPOCHS
# --- hyperparameter grid (arch fixed = v2), targeting the epoch-1 overfit ---
CONFIGS=(
  "v2_base|v2|5e-5|5|1e-2|0.3|64|30"          # reference = current v2 config
  "v2_lr2e5|v2|2e-5|5|1e-2|0.3|64|30"         # slower LR
  "v2_lr1e5|v2|1e-5|5|1e-2|0.3|64|30"         # slowest LR
  "v2_lr1e4|v2|1e-4|5|1e-2|0.3|64|30"         # faster LR (best was at TOP of range — bracket up)
  "v2_lr2e4|v2|2e-4|5|1e-2|0.3|64|30"         # faster still
  "v2_lr2e5_wu10|v2|2e-5|10|1e-2|0.3|64|40"   # slow LR + long warmup + more epochs
  "v2_wd5e2|v2|5e-5|5|5e-2|0.3|64|30"         # stronger weight decay
  "v2_do05|v2|5e-5|5|1e-2|0.5|64|30"          # stronger dropout
  "v2_h32|v2|5e-5|5|1e-2|0.3|32|30"           # smaller model (small N)
  "v2_h32_reg|v2|2e-5|5|5e-2|0.5|32|30"       # small + heavy reg + slow
  "v2_slow|v2|1e-5|10|1e-2|0.3|64|50"         # slow schedule, long horizon
  "v2_combo|v2|2e-5|5|5e-2|0.5|64|30"         # combined regularization
  # --- arch sanity check (baseline HP, does the ranking survive family_id CV?) ---
  "v1_base|v1|5e-5|5|1e-2|0.3|64|30"
  "v3_base|v3|5e-5|5|1e-2|0.3|64|30"
  "v4_base|v4|5e-5|5|1e-2|0.3|64|30"
)

n=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME ARCH LR WU WD DO HID EP <<< "$cfg"
  for ((f=0; f<NUM_FOLDS; f++)); do
    SAVE="$OUTROOT/${NAME}_fold${f}"
    LOG="$LOGDIR/hpsweep_${NAME}_fold${f}_%J.log"
    # Idempotent re-runs: skip a fold that already produced history.json so
    # adding configs doesn't re-submit (or clobber) the ones already done.
    # FORCE=1 to re-run anyway.
    if [ "${FORCE:-0}" != "1" ] && [ -f "$SAVE/history.json" ]; then
      continue
    fi
    n=$((n+1))
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry] ${NAME} fold${f}: arch=$ARCH lr=$LR wu=$WU wd=$WD do=$DO hid=$HID ep=$EP -> $SAVE"
    else
      env MM_ARCH="$ARCH" \
          MM_VIDEO_FEATURE_DIR="$VIDEO_DIR" MM_AUDIO_FEATURE_DIR="$AUDIO_DIR" \
          MM_SAVE_DIR="$SAVE" \
          MM_GROUP_KEY=family_id MM_NUM_FOLDS="$NUM_FOLDS" MM_FOLD_IDX="$f" \
          MM_LEARNING_RATE="$LR" MM_WARMUP_EPOCHS="$WU" MM_WEIGHT_DECAY="$WD" \
          MM_DROPOUT="$DO" MM_VIDEO_HIDDEN="$HID" MM_AUDIO_HIDDEN="$HID" MM_HEAD_HIDDEN="$HID" \
          MM_EPOCHS="$EP" \
          bsub -M "$MEM_MB" -R "$RES" -oo "$LOG" < "$TRAIN_BSUB"
    fi
  done
done

echo "=== ${#CONFIGS[@]} configs x $NUM_FOLDS folds = $n jobs $([ "$DRY_RUN" = 1 ] && echo '(DRY-RUN — set DRY_RUN=0 to submit)') ==="
echo "Results:  $OUTROOT/<config>_fold<k>/history.json"
echo "Summarize: python scripts/summarize_hp_sweep.py $OUTROOT"

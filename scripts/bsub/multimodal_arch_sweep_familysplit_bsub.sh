#!/bin/bash
# Launcher: architecture-lever sweep on the leak-free family_id metric, plus the
# proper full-dataset (merged 89) family_id CV. Everything is 5-fold family_id CV
# so results are mean+/-SE and directly comparable to the HP sweep.
#
# WHY: after the subject_id->family_id leakage fix, HP tuning was flat (~0.714),
# so the remaining lever is ARCHITECTURE with the right inductive bias for a
# *relational* task (synchrony = coupling between two people over time). We test:
#   - v6_crossperson: per-person features + symmetric cross-person attention
#     (the architecturally-correct relational lever; only ever tested on the
#     leaky split before -> this is its first clean eval). Original-49 only
#     (per-person features don't exist for EmoGrow/R56).
#   - v7_temporal_context: multi-second context (synchrony has >1s dynamics).
#   - v8_subjinv: v6_crossperson + adversarial subject head (identity-invariance)
#     on per-person features. v2_subjinv: the SAME adversarial head on the clean
#     2D features + v2 baseline -- the feature-quality-decoupled test of whether
#     removing dyad identity raises within-recording AUC (the confound we found).
#   - v2_pose: pose cue.
#   - full-data merged v2/v8: does the new data help on a real CV estimate.
# HP held at the v2 baseline (script defaults) so only architecture varies.
# See feedback_multimodal_family_split_leak, project_multimodal_ceiling_investigation.
#
# LAUNCHER (run directly, NOT via `bsub <`). Submits one job per (config, fold)
# by reusing scripts/bsub/multimodal_from_features_bsub.sh (v19) with per-job env.
#
# Usage (dry-run FIRST, then submit):
#   bash scripts/bsub/multimodal_arch_sweep_familysplit_bsub.sh            # prints only
#   DRY_RUN=0 bash scripts/bsub/multimodal_arch_sweep_familysplit_bsub.sh  # submits
SCRIPT_VERSION="multimodal_arch_sweep_familysplit-v2"
echo "=== [$SCRIPT_VERSION] ==="

set -u
SYNCHRONAI_DIR="${SYNCHRONAI_DIR:-/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI}"
cd "$SYNCHRONAI_DIR" || { echo "ERROR: cannot cd $SYNCHRONAI_DIR"; exit 1; }

TRAIN_BSUB="scripts/bsub/multimodal_from_features_bsub.sh"
LOGDIR="scripts/bsub/logs"
OUTROOT="runs/multimodal_features/arch_sweep_familysplit"
DRY_RUN="${DRY_RUN:-1}"
NUM_FOLDS="${NUM_FOLDS:-5}"
mkdir -p "$LOGDIR"

# name | ARCH | VIDEO_DIR | AUDIO_DIR | MEM_GB | EXTRA(space-sep MM_* or '-')
# HP is intentionally NOT set here -> inherits the v2 baseline defaults in the
# training script (lr 5e-5, wd 1e-2, dropout 0.3, hidden 64, warmup 5, ep 30).
CONFIGS=(
  # --- architecture levers, ORIGINAL 49 (honest metric, comparable to HP sweep) ---
  "arch_v2_orig|v2|data/dinov2_features_meanpatch|data/wavlm_baseplus_features|24|-"
  "arch_v6crossperson_orig|v6_crossperson|data/perperson_video_features|data/wavlm_baseplus_features|48|-"
  # Option 1: v8 = v6_crossperson + adversarial head. Needs per-person 3D features
  # (same dir as arch_v6crossperson_orig, so v8-vs-v6 is a clean A/B for the head).
  # The earlier v8 runs crashed feeding it 2D meanpatch features.
  "arch_v8subjinv_l01_orig|v8_subjinv|data/perperson_video_features|data/wavlm_baseplus_features|48|MM_LAMBDA_SUBJ=0.1"
  "arch_v8subjinv_l03_orig|v8_subjinv|data/perperson_video_features|data/wavlm_baseplus_features|48|MM_LAMBDA_SUBJ=0.3"
  # Option 2: v2_subjinv = same adversarial subject-invariance head on the CLEAN
  # 2D features + v2 baseline. Isolates the identity-invariance lever from v6's
  # per-person feature-quality problems -- the fair test of the confound lever.
  "arch_v2subjinv_l01_orig|v2_subjinv|data/dinov2_features_meanpatch|data/wavlm_baseplus_features|24|MM_LAMBDA_SUBJ=0.1"
  "arch_v2subjinv_l03_orig|v2_subjinv|data/dinov2_features_meanpatch|data/wavlm_baseplus_features|24|MM_LAMBDA_SUBJ=0.3"
  "arch_v2pose_orig|v2_pose|data/dinov2_features_meanpatch|data/wavlm_baseplus_features|24|MM_POSE_FEATURE_DIR=data/pose_features"
  # v7 context stacks multiple seconds -> larger preload; generous mem + verify one fold first.
  "arch_v7ctx2_orig|v7_temporal_context|data/dinov2_features_meanpatch|data/wavlm_baseplus_features|64|MM_CONTEXT_WINDOW=2"
  # --- full merged dataset (89), does-new-data-help on a real CV estimate ---
  "full_v2_merged|v2|data/dinov2_features_meanpatch_plus|data/wavlm_baseplus_features_plus|48|-"
  # v8_subjinv can't run on merged data (per-person features don't exist for
  # EmoGrow/R56); v2_subjinv runs on the 2D merged features -> subject-invariance
  # on the full 89, comparable to full_v2_merged.
  "full_v2subjinv_l01_merged|v2_subjinv|data/dinov2_features_meanpatch_plus|data/wavlm_baseplus_features_plus|48|MM_LAMBDA_SUBJ=0.1"
  # --- feature-side levers on the ORIGINAL 49 (the untouched part of the model) ---
  # #1 WavLM-large (1024-dim, already extracted, drop-in). #2 input LayerNorm on
  # raw frozen features. Plus the combination. v2, baseline HP, family_id CV.
  "feat_wavlmlarge_orig|v2|data/dinov2_features_meanpatch|data/wavlm_large_features|32|-"
  "feat_inputnorm_orig|v2|data/dinov2_features_meanpatch|data/wavlm_baseplus_features|24|MM_INPUT_NORM=1"
  "feat_inputnorm_wavlmlarge_orig|v2|data/dinov2_features_meanpatch|data/wavlm_large_features|32|MM_INPUT_NORM=1"
)

n=0; skipped=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME ARCH VDIR ADIR MEMGB EXTRA <<< "$cfg"
  [ "$EXTRA" = "-" ] && EXTRA=""
  # Skip (don't kill the whole launch) if this config's feature dirs are absent.
  if [ ! -f "$VDIR/feature_index.csv" ] || [ ! -f "$ADIR/feature_index.csv" ]; then
    echo "SKIP $NAME: missing $VDIR or $ADIR feature_index.csv"
    skipped=$((skipped+1)); continue
  fi
  for ((f=0; f<NUM_FOLDS; f++)); do
    SAVE="$OUTROOT/${NAME}_fold${f}"
    LOG="$LOGDIR/archsweep_${NAME}_fold${f}_%J.log"
    # Idempotent re-runs: skip folds that already have history.json so adding
    # configs doesn't re-submit the ones already done. FORCE=1 to override.
    if [ "${FORCE:-0}" != "1" ] && [ -f "$SAVE/history.json" ]; then
      continue
    fi
    RES="select[mem>${MEMGB}GB] rusage[mem=${MEMGB}GB] span[hosts=1]"
    n=$((n+1))
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry] $NAME fold$f: arch=$ARCH video=$VDIR mem=${MEMGB}GB extra='${EXTRA:-none}' -> $SAVE"
    else
      env MM_ARCH="$ARCH" MM_VIDEO_FEATURE_DIR="$VDIR" MM_AUDIO_FEATURE_DIR="$ADIR" \
          MM_SAVE_DIR="$SAVE" MM_GROUP_KEY=family_id MM_NUM_FOLDS="$NUM_FOLDS" MM_FOLD_IDX="$f" \
          $EXTRA \
          bsub -M "${MEMGB}000000" -R "$RES" -oo "$LOG" < "$TRAIN_BSUB"
    fi
  done
done

echo "=== submitted $n jobs, skipped $skipped configs $([ "$DRY_RUN" = 1 ] && echo '(DRY-RUN — set DRY_RUN=0 to submit)') ==="
echo "Results:   $OUTROOT/<config>_fold<k>/history.json"
echo "Summarize: python scripts/summarize_hp_sweep.py $OUTROOT"

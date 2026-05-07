#!/bin/bash
SCRIPT_VERSION="multimodal_from_features_bsub-v10"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 24000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 4
#BSUB -R 'select[mem>24GB] rusage[mem=24GB] span[hosts=1]'

# CPU-only multi-modal training on pre-extracted DINOv2 + WavLM features.
# Mirrors the wavlm_audio_sweep_bsub.sh pattern: ml-env is maintained by
# extraction scripts, this script just uses the absolute python path.
#
# v5 changes (defaults target v2-baseline arch):
#   - span[hosts=1]: keep all -n 8 slots on one host (v1 fragmented across 4)
#   - OMP/MKL thread exports: PyTorch CPU pool defaults to host cores, not
#     LSF -n; without these the slots are wasted regardless of span.
#   - --arch v2: select MultiModalV2 model (projection + 2-layer LSTM +
#     explicit aggregator dropout + concat). Override with MM_ARCH=v1 to
#     reproduce v1 runs.
#   - New default LR (5e-5), WD (1e-2), batch (128), epochs (30),
#     patience (10), save_dir (v2_baseline) per docs/multimodal_v2_plan.md.

ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

cd $SYNCHRONAI_DIR

export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"

# Tell PyTorch to use the LSF slot count. span[hosts=1] alone is not enough.
# v7: dropped 8 → 4 to match -n 4. v2_baseline run used only ~1.7 effective
# cores time-averaged anyway (LSTM is sequential), so 4 is enough headroom for
# spikes and frees slots for other users.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

ARCH="${MM_ARCH:-v2}"
VIDEO_FEATURE_DIR="${MM_VIDEO_FEATURE_DIR:-data/dinov2_features_meanpatch}"
AUDIO_FEATURE_DIR="${MM_AUDIO_FEATURE_DIR:-data/wavlm_baseplus_features}"
SAVE_DIR="${MM_SAVE_DIR:-runs/multimodal_features/v2_baseline}"
EPOCHS="${MM_EPOCHS:-30}"
BATCH_SIZE="${MM_BATCH_SIZE:-128}"
LEARNING_RATE="${MM_LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${MM_WEIGHT_DECAY:-1e-2}"
DROPOUT="${MM_DROPOUT:-0.3}"
VIDEO_HIDDEN="${MM_VIDEO_HIDDEN:-64}"
AUDIO_HIDDEN="${MM_AUDIO_HIDDEN:-64}"
HEAD_HIDDEN="${MM_HEAD_HIDDEN:-64}"
WARMUP_EPOCHS="${MM_WARMUP_EPOCHS:-5}"
PATIENCE="${MM_PATIENCE:-10}"
NUM_WORKERS="${MM_NUM_WORKERS:-0}"
# v6: early stopping on val_loss for v2 (val_auc peaks at warmup epoch 1, biases
# best.pt to undertrained ckpt). Override with MM_EARLY_STOP_METRIC=val_auc to
# reproduce v1 behavior exactly.
EARLY_STOP_METRIC="${MM_EARLY_STOP_METRIC:-val_loss}"
# v9: optional k-fold CV. Default num_folds=0 → original 80/20 split.
NUM_FOLDS="${MM_NUM_FOLDS:-0}"
FOLD_IDX="${MM_FOLD_IDX:--1}"
# v10: single-modality ablation. 'both' (default) = full multimodal,
# 'video' = audio zeroed out, 'audio' = video zeroed out.
MODALITY="${MM_MODALITY:-both}"

echo "=== [$SCRIPT_VERSION] ==="
echo "  Arch:           $ARCH"
echo "  Video features: $VIDEO_FEATURE_DIR"
echo "  Audio features: $AUDIO_FEATURE_DIR"
echo "  Save dir:       $SAVE_DIR"
echo "  Epochs:         $EPOCHS  (warmup: $WARMUP_EPOCHS, patience: $PATIENCE)"
echo "  Batch size:     $BATCH_SIZE"
echo "  LR / WD:        $LEARNING_RATE / $WEIGHT_DECAY"
echo "  Dropout:        $DROPOUT"
echo "  Hidden dims:    video=$VIDEO_HIDDEN audio=$AUDIO_HIDDEN head=$HEAD_HIDDEN"
echo "  Early-stop on:  $EARLY_STOP_METRIC"
if [ "$NUM_FOLDS" -gt 0 ]; then
    echo "  K-fold CV:      fold $FOLD_IDX of $NUM_FOLDS"
fi
if [ "$MODALITY" != "both" ]; then
    echo "  Modality:       $MODALITY (other modality zeroed for ablation)"
fi
echo "  Threads:        OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"

echo "=== Preflight: checking ml-env imports ==="
"$ML_PY" -c "import torch, pandas, sklearn, matplotlib; print(f'torch={torch.__version__} pandas ok sklearn ok matplotlib ok')" || {
    echo "ERROR: ml-env missing required packages."
    exit 1
}

if [ ! -f "$VIDEO_FEATURE_DIR/feature_index.csv" ]; then
    echo "ERROR: Video features not found at $VIDEO_FEATURE_DIR/feature_index.csv"
    exit 1
fi
if [ ! -f "$AUDIO_FEATURE_DIR/feature_index.csv" ]; then
    echo "ERROR: Audio features not found at $AUDIO_FEATURE_DIR/feature_index.csv"
    exit 1
fi

echo ""
echo "=== Starting Multi-Modal Feature Training ==="

"$ML_PY" scripts/train_multimodal_from_features.py \
    --arch "$ARCH" \
    --video-feature-dir "$VIDEO_FEATURE_DIR" \
    --audio-feature-dir "$AUDIO_FEATURE_DIR" \
    --save-dir "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    --video-hidden "$VIDEO_HIDDEN" \
    --audio-hidden "$AUDIO_HIDDEN" \
    --head-hidden "$HEAD_HIDDEN" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --patience "$PATIENCE" \
    --num-workers "$NUM_WORKERS" \
    --early-stop-metric "$EARLY_STOP_METRIC" \
    --num-folds "$NUM_FOLDS" \
    --fold-idx "$FOLD_IDX" \
    --modality "$MODALITY"
train_rc=$?
if [ $train_rc -ne 0 ]; then
    echo "ERROR: training exited with code $train_rc — see traceback above."
    echo "  Multimodal training FAILED, no completion banner."
    exit $train_rc
fi

echo ""
echo "=== Done ==="
echo "Results: $SAVE_DIR/history.json"

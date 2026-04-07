# synchronAI

Multi-modal AI system for predicting dyadic synchrony (coordinated behavior between
two people) using video, audio, and fNIRS brain signals.

## Architecture

Three independent modality pipelines feed into a fusion model:

```
Video  (DINOv2-small) ──→ temporal LSTM ──→ features ──┐
Audio  (WavLM-base+)  ──→ projection   ──→ features ──┼──→ fusion ──→ synchrony
fNIRS  (DDPM U-Net)   ──→ per-pair enc ──→ features ──┘
```

Each pipeline can run independently or be combined at the fusion level. Features
are pre-extracted to disk for fast downstream training.

## Modality Pipelines

### Video
- **Backbone**: DINOv2-small (384-dim, 224x224 input)
- **Temporal**: LSTM or attention over frame sequences
- **Training**: Two-stage fine-tuning (frozen backbone, then differential LR unfreeze)
- **Features**: Pre-extracted at multiple resolutions (112, 168, 224)

### Audio
- **Backbone**: WavLM-base-plus (768-dim per layer, 16kHz input)
- **Features**: Per-layer extraction for learned layer weighting
- **Supports**: Audio files (WAV, FLAC) and video files (MP4, MOV via ffmpeg)

### fNIRS
- **Generative**: DDPM diffusion model learns HbO/HbR hemodynamic dynamics
- **Architecture**: Per-pair (feature_dim=2) — one source-detector pair at a time
- **Transfer**: Frozen encoder features → child/adult classifier → synchrony
- **Quality control**: Multi-stage QC with tiered data quality (gold/standard/salvageable)
- **Sweep**: 4 encoder sizes (micro/small/medium/large) x 5 classifier architectures

### Multimodal Fusion
- **Strategies**: Concat, gated, or temporal cross-attention
- **Cross-attention**: Operates on frame-level sequences (B, T, D), not pooled vectors

## Quick Start

```bash
# Install (editable, for development)
pip install -e .

# Or with optional audio dependencies
pip install -e ".[audio]"
```

### Train a video synchrony classifier
```bash
python -m synchronai.main --video --train classifier \
    --data-dir path/to/labels.csv \
    --save-dir runs/video_classifier \
    --backbone dinov2-small \
    --temporal-aggregation lstm
```

### Train fNIRS diffusion model (per-pair)
```bash
bash scripts/generative_pretrain.sh \
    --save-dir runs/fnirs_perpair \
    --per-pair \
    --unet-base-width 32 \
    --enable-qc --sci-threshold 0.40 --snr-threshold 2.0
```

### Extract features from pretrained encoder
```bash
python scripts/extract_fnirs_features.py \
    --encoder-weights runs/fnirs_perpair/fnirs_unet_encoder.pt \
    --data-dirs "/path/to/NIRS_data" \
    --output-dir data/fnirs_features \
    --per-pair --enable-qc \
    --include-tiers "gold,standard,salvageable"
```

### Train child/adult classifier on extracted features
```bash
python scripts/train_fnirs_from_features.py \
    --feature-dir data/fnirs_features \
    --save-dir runs/classifier \
    --pool lstm --hidden-dim 64 \
    --include-tiers "gold,standard" \
    --holdout-tiers "gold,salvageable"
```

## Project Structure

```
src/synchronai/
    models/          # Architectures (video, audio, fNIRS, multimodal)
    training/        # Training loops per modality
    data/            # Datasets, preprocessing, quality control
    inference/       # Prediction and generation
    evaluation/      # IRR analysis, metrics
    utils/           # Logging, wandb, visualization, config

scripts/
    *.py             # Feature extraction, training, analysis utilities
    bsub/            # LSF cluster submission scripts (versioned)
    bsub/archive/    # Deprecated scripts

docs/
    plans/           # Transfer learning plan, upgrade roadmap
    *.md             # Architecture docs, results, troubleshooting

runs/                # Training outputs (gitignored)
data/                # Extracted features, labels (gitignored)
```

## Cluster (LSF/BSub)

All BSub scripts live in `scripts/bsub/` and include `SCRIPT_VERSION` for
log traceability. See [CLAUDE.md](CLAUDE.md) for cluster conventions and
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

```bash
# Submit fNIRS per-pair pretraining (4 model sizes)
sh scripts/bsub/pre_fnirs_perpair_pretrain_bsub.sh

# Submit child/adult classification sweep (20 jobs)
sh scripts/bsub/pre_fnirs_child_adult_sweep_bsub.sh
```

## Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Claude Code instructions, cluster conventions |
| [AGENTS.md](AGENTS.md) | Development principles, workflows |
| [Transfer Learning Plan](docs/plans/fnirs_transfer_plan.md) | fNIRS pipeline roadmap (Phases 1-5) |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Cluster and training debugging |
| [Methods & Results](docs/fnirs_generative_methods_results.md) | fNIRS generative pretraining paper-style writeup |
| [Transfer Learning Fixes](docs/transfer_learning_fixes.md) | Critical bugs fixed during multimodal integration |
| [Multimodal Heatmaps](docs/multimodal_heatmaps.md) | Visualization of fusion predictions |

## Key Design Decisions

- **Subject-grouped splits**: All train/val splits group by subject_id to prevent data leakage
- **Per-pair fNIRS**: Universal HbO/HbR dynamics, generalizes to any montage configuration
- **Quality tiers**: Gold (pristine) / standard / salvageable — holdout evaluation on each tier during training
- **Pre-extracted features**: Frozen backbone features saved to disk for fast classifier sweeps
- **No pip install in cluster jobs**: Use PYTHONPATH to avoid NFS race conditions

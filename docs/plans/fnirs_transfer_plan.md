# fNIRS Diffusion Transfer Learning Plan

*Created: 2026-03-26, Updated: 2026-03-27*

## Overview

The fNIRS DDPM U-Net (v3, TensorFlow) has converged at epoch 621 with best val loss
0.003452 at epoch 491. This plan uses the pretrained encoder as a frozen feature
extractor through a **three-stage transfer learning pipeline**:

```
Stage A: Child vs Adult Classifier (standalone model with sweep)
    Pretrained U-Net encoder → temporal model → child/adult prediction
    This learns "who is this brain?" from hemodynamic signatures.

Stage B: Transfer to Synchrony
    Freeze child/adult model as feature extractor
    Penultimate activations → synchrony classification head
    "Who is this brain?" → "Are these two brains in sync?"

Stage C: Multimodal Fusion
    Video (DINOv2) + Audio (WavLM) + fNIRS (pretrained head)
    Three-way fusion → synchrony prediction
```

### Core Challenges

1. **TF→PyTorch bridge**: Diffusion model is TensorFlow; rest is PyTorch.
   Replicate encoder in PT, convert weights.

2. **Time scale**: fNIRS v3 uses 60s windows at 7.8Hz (472 padded timesteps).
   After 3 levels of stride-2 downsampling, bottleneck has 59 timesteps — roughly
   1 per second (~1.02s/timestep).

3. **Small dataset**: ~100-200 recordings across CARE + P-CAT. Multi-minute
   recordings are windowed into 60s segments to get ~500-2000 samples.

4. **Child/adult label detection** (confirmed from directory exploration):

   | Study | Child indicator | Parent indicator | Example |
   |-------|----------------|------------------|---------|
   | CARE | 5-digit ID in dir name | 4-digit ID in dir name | `50001_V0_fNIRS/` vs `5000_V0_fNIRS/` |
   | P-CAT R56 | `-C_` in dir name | `-P_` in dir name | `1102-C_fNIRS_DB-DOS/` |
   | P-CAT R01 | `_C` suffix on dir | `_P` suffix on dir | `11001_C/` or `12001_P/` |

   All recordings are **NIRx directories** (contain `.hdr` files). CARE also
   has `.snirf` files but NIRx dirs are primary. Discovery avoids double-counting.

---

## Phase 1: PyTorch U-Net Encoder + Weight Conversion (DONE)

### 1.1 Encoder Replica

**File**: `src/synchronai/models/fnirs/unet_encoder_pt.py` — **Done**

```
Input: (B, 472, 20) — 60s at 7.8Hz, padded from 469
Encoder:
  Level 0: ResBlock(20→64) + Conv1D(s=2) → (B, 236, 64)
  Level 1: ResBlock(64→128) + Conv1D(s=2) → (B, 118, 128)
  Level 2: ResBlock(128→256) + Conv1D(s=2) → (B, 59, 256)
Bottleneck: 2× ResBlock(256→512) → (B, 59, 512)
```

Supports multi-scale extraction: `return_all_levels=True` returns all encoder
levels + bottleneck (total 960-dim when mean-pooled: 64+128+256+512).

### 1.2 Weight Conversion

**File**: `scripts/convert_fnirs_tf_to_pt.py` — **Done**

Converts TF Keras weights to PT state dict with:
- Conv1D transpose: `(K, C_in, C_out)` → `(C_out, C_in, K)`
- Dense transpose: `(in, out)` → `(out, in)`
- `--verify` flag compares TF vs PT bottleneck output (atol < 1e-4)

13 Conv1D + 7 Dense + 10 LayerNorm in encoder path.

---

## Phase 2: Feature Extraction with Windowing (DONE / needs update)

### 2.1 Windowed Extraction

**File**: `scripts/extract_fnirs_features.py` — **Done (needs windowing update)**

Each multi-minute fNIRS recording is sliced into 60s windows:
- **Non-overlapping** (default): 10-min recording → 10 windows
- **30s-stride overlap** (variant): 10-min recording → 19 windows
- Each window → encoder → `(59, 512)` bottleneck features saved as `.pt`
- All windows from the same recording get the same child/adult label

### 2.2 Child/Adult Label Detection

Built into extraction script. Label rules:
- **P-CAT**: filename contains `C` → child, `P` → parent
- **CARE**: subject ID length 5 digits → child, 4 digits → parent
- Override with `--participant-labels` CSV

### 2.3 Normalization

Uses `feature_mean`/`feature_std` from `fnirs_diffusion_config.json` directly
(not the Welford state in `running_stats.npz`).

### 2.4 Data Sources

All recordings from both studies used for child/adult classification:
- **CARE**: `/storage1/.../CARE/NIRS_data/`
- **P-CAT R56**: `/storage1/.../P-CAT/R56/NIRS_data/`
- **P-CAT R01**: 6 directories (PSU/WUSTL × T1/T3/T5)

---

## Phase 3: Child/Adult Classifier Sweep (Stage A)

**Goal**: Train a strong child vs adult classifier — not a sanity check but a
real model whose learned representations transfer to synchrony.

### 3.1 Sweep Design

~100-200 recordings × ~5-10 windows = **500-2000 samples**. Small dataset →
aggressive regularization, smaller architectures.

**Feature variants** (2):
- Bottleneck only: (59, 512) per window
- Multiscale: mean-pooled all levels (960-dim vector) per window

**Architecture variants** (5 per feature type):

| Run | Temporal | Hidden | Dropout | Project | Notes |
|-----|----------|--------|---------|---------|-------|
| linear_probe | mean pool | 0 | 0 | 0 | Baseline: Linear(512→1) |
| mlp_small | mean pool | 32 | 0.3 | 0 | Small MLP |
| mlp_proj128 | mean pool | 64 | 0.5 | 128 | Projection bottleneck |
| lstm_small | LSTM | 64 | 0.3 | 0 | Temporal modeling |
| lstm_proj128 | LSTM | 64 | 0.5 | 128 | LSTM + projection |

**Overlap variant** (1):
- Best architecture from above re-run with 30s-stride overlapping windows

**Total sweep**: ~11 jobs (5 bottleneck + 5 multiscale + 1 overlap)

### 3.2 Training Config

- Optimizer: AdamW, LR=1e-3 (linear probe) / 3e-4 (MLP/LSTM)
- Schedule: warmup 3 epochs + cosine decay
- Weight decay: 1e-2
- Early stopping: patience=15 on val AUC
- Split: subject-grouped 80/20 (critical — same subject's windows must stay together)
- Epochs: 50 max
- Batch size: 32

### 3.3 Success Criteria

- **Linear probe AUC > 0.6**: Encoder learned basic hemodynamic structure
- **Best sweep AUC > 0.75**: Representations are rich enough to transfer
- **Train-val gap < 15%**: Not severely overfitting

The best model from this sweep becomes the **fNIRS feature extractor** for
Stage B (synchrony) and Stage C (fusion).

---

## Phase 4: Transfer to Synchrony (Stage B)

Only pursue after Phase 3 produces a strong child/adult classifier.

### 4.1 Feature Extraction from Classifier

Freeze the best child/adult model. Use its **penultimate activations**
(the layer before the final Linear) as features for synchrony:

```python
# For MLP: output of the ReLU before final Linear → (B, hidden_dim)
# For LSTM: LSTM hidden state → (B, lstm_hidden_dim)
```

This produces a compact, learned representation that encodes "what kind of
brain signal is this?" — foundational for synchrony prediction.

### 4.2 Per-Second Synchrony Features

For synchrony, we need per-second features aligned to video/audio labels:
1. Extract per-second features from the child/adult model's penultimate layer
2. Match to `labels.csv` via session mapping (`data/session_mapping.csv`)
3. Save per-second features following audio/video extraction pattern

### 4.3 Session Mapping (Video ↔ fNIRS)

**Create**: `src/synchronai/data/multimodal/session_mapping.py`
**Create**: `data/session_mapping.csv`

```csv
video_path,fnirs_path,subject_id,session,fnirs_offset_seconds
/path/to/video.mp4,/path/to/fnirs.snirf,50001,V0,0.0
```

Only CARE data has both video and fNIRS. P-CAT is fNIRS-only.

---

## Phase 5: Three-Way Multimodal Fusion (Stage C)

Only pursue after Phase 4 shows fNIRS adds signal beyond video+audio.

### 5.0 Architecture: Separate Pipelines, Joined by Subject ID

Each modality has its own independent extraction pipeline. They connect
at fusion time via `subject_id` as the natural join key:

```
Video pipeline:  raw video   → DINOv2 → feature_index.csv (subject_id, second)
Audio pipeline:  raw video   → WavLM  → feature_index.csv (subject_id, second)
fNIRS pipeline:  raw NIRx    → U-Net  → feature_index.csv (subject_id, window_idx)
                                              │
                            JOIN on (subject_id, second)
                                              │
                                    Trimodal feature dataset
```

This design means:
- Each modality runs independently — no cross-dependencies during extraction
- You can re-run one modality without touching others
- Subject IDs are already consistent across modalities in CARE:
  - Video: `50001_V0_DB-DOS.mp4` → subject_id = `50001`
  - fNIRS: `50001_V0_fNIRS/` → subject_id = `50001`
- The `session_mapping.csv` confirms pairings and handles temporal offsets

### 5.1 Extend Fusion Modules to N Modalities

**Modify**: `src/synchronai/models/multimodal/fusion_modules.py`

Update `create_fusion_module()` to accept N modality dimensions.

### 5.2 Trimodal Training

All-features-on-disk: no encoders loaded at train time. Join video + audio +
fNIRS features by `(subject_id, second)`. Handle missing modalities with
zero-padding + modality mask.

---

## Implementation Order

```
Phase 1 (PT encoder) ──── DONE
         │
Phase 2 (extraction)  ──── DONE (needs windowing update)
         │
Phase 3 (child/adult sweep) ──── NEXT
         │
         ├── if AUC > 0.75 ──→ Phase 4 (synchrony transfer)
         │                              │
         │                     Phase 5 (trimodal fusion)
         │
         └── if AUC < 0.6 ──→ Fallbacks:
                                - Multiscale features
                                - Features at t > 0
                                - Direct transformer (no DDPM)
```

---

## Pipeline Commands (Run in Order)

### Step 1: Convert weights (~2 min, 4GB RAM)
```bash
python scripts/convert_fnirs_tf_to_pt.py \
    --config-json runs/fnirs_diffusion_v3/fnirs_diffusion_config.json \
    --weights-path runs/fnirs_diffusion_v3/fnirs_unet.weights.h5 \
    --verify
```

### Step 2: Extract features (~30 min, 4GB RAM)
```bash
python scripts/extract_fnirs_features.py \
    --encoder-weights runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt \
    --data-dirs "/storage1/.../CARE/NIRS_data/:/storage1/.../P-CAT/R56/NIRS_data/" \
    --output-dir data/fnirs_encoder_features
```

### Step 2b: Extract multiscale features
```bash
python scripts/extract_fnirs_features.py \
    --encoder-weights runs/fnirs_diffusion_v3/fnirs_unet_encoder.pt \
    --data-dirs "/storage1/.../CARE/NIRS_data/:/storage1/.../P-CAT/R56/NIRS_data/" \
    --output-dir data/fnirs_multiscale_features \
    --multiscale
```

### Step 3: Run child/adult sweep (~1 hour total, <1GB RAM each)
```bash
sh scripts/bsub/pre_fnirs_child_adult_sweep_bsub.sh
```

---

## Files Summary

| File | Action | Phase | Status |
|------|--------|-------|--------|
| `src/synchronai/models/fnirs/unet_encoder_pt.py` | Create | 1 | **Done** |
| `scripts/convert_fnirs_tf_to_pt.py` | Create | 1 | **Done** |
| `scripts/extract_fnirs_features.py` | Create | 2 | **Done** (needs windowing) |
| `src/synchronai/data/fnirs/feature_dataset.py` | Create | 3 | **Done** |
| `scripts/train_fnirs_from_features.py` | Create | 3 | **Done** |
| `scripts/bsub/pre_fnirs_child_adult_sweep_bsub.sh` | Create | 3 | Pending |
| `scripts/bsub/fnirs_child_adult_sweep_bsub.sh` | Create | 3 | Pending |
| `src/synchronai/data/multimodal/session_mapping.py` | Create | 4 | Pending |
| `data/session_mapping.csv` | Create | 4 | Pending |
| `src/synchronai/models/multimodal/fusion_modules.py` | Modify | 5 | Pending |
| `src/synchronai/data/multimodal/trimodal_feature_dataset.py` | Create | 5 | Pending |
| `scripts/train_trimodal_from_features.py` | Create | 5 | Pending |

---

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Weight mapping errors (TF→PT) | Garbage features | `--verify` flag, atol < 1e-4 |
| Features at t=0 not discriminative | Sweep AUC ≈ 0.5 | Fallback to t>0, multiscale |
| Small dataset (~500-2000 windows) | Overfitting | Aggressive regularization, subject-grouped splits, small architectures |
| Child/adult not separable from fNIRS alone | Model can't learn | Would indicate encoder didn't learn hemodynamic structure → reconsider approach |
| Cross-study label inconsistency | Wrong labels | Per-study parsing with explicit rules + manual spot-check |
| Correlated windows from same recording | Inflated metrics | Subject-grouped splits ensure all windows from same subject stay together |

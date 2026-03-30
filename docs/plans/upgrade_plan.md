# synchronAI Upgrade Plan

## Current State Summary

*Updated: 2026-03-26*

The system predicts binary dyadic synchrony (sync/async) at 1-second resolution from
adult-child interaction video. Three modality pipelines exist:

| Modality | Backbone | Framework | Status |
|----------|----------|-----------|--------|
| Video | DINOv2-small → LSTM → MLP | PyTorch | Sweep done (best AUC 0.697). Multi-rez stage 1 running. |
| Audio | WavLM-base-plus (768-dim, 12 layers) | PyTorch | Sweep v1 done (AUC ~0.678). Sweep v2 (per-layer + projection) submitted. |
| fNIRS | 1D U-Net DDPM (base_width=64, cosine schedule) | TensorFlow | Converged (epoch 621, best val loss 0.003452). Transfer plan ready. |
| Multimodal | Video + Audio temporal cross-attention fusion | PyTorch | Working (cross-attention fix applied) |

**Dataset**: 59,250 labeled seconds across subjects 50001-50581, all session V0.
Class distribution: ~44% async (0) / ~56% sync (1) -- mild imbalance.
Labels derived from CARE synchrony study Excel files with multiple annotators per session.
Audio is single-microphone (no separate mics per person).

---

## Phase 0: Critical Bug Fixes (Do First)

These are bugs in the current codebase that must be fixed before any upgrades.

### 0.1 Cross-Attention Fusion Is a No-Op -- DONE

**Status**: Fixed (2026-03-23). `CrossModalAttention` now operates on temporal token
sequences `(B, T_v, D)` and `(B, T_a, D)` with proper cross-attention, residual
connections, and LayerNorm. Pooling happens after attention. The fusion model routes
frame-level features (video `frame_features`, audio `sequence_features`) to cross-attention
while concat/gated fusion continues to use pooled vectors.

### 0.2 Audio Event Auxiliary Labels -- DONE

**Status**: Fixed. The audio dataset properly maps event labels to all 7 classes (0-6).
The multimodal training loop sets `event_loss_weight=0` by default, safely disabling the
auxiliary task when labels aren't available. Re-enable by setting `event_loss_weight > 0`
when real event labels exist.

### 0.3 Whisper `.to()` Method -- DONE

**Status**: Both bugs fixed. Encoder is now registered as `self.encoder` (not `_whisper`),
and gradient context is determined dynamically via `any(p.requires_grad for p in self.encoder.parameters())`
instead of the stale `_freeze` flag. Stage 2 fine-tuning works correctly.

### 0.4 Whisper Lazy Loading Defeats `state_dict()` Round-Tripping -- DONE

**Status**: Fixed. `load_state_dict()` override calls `_load_model()` before restoring
weights, ensuring encoder is populated before key matching.

### 0.5 `create_multimodal_splits` Mutates Shared Configs

**File**: `src/synchronai/data/multimodal/dataset_mm.py:336-364`

`video_config` and `audio_config` are modified in place (setting `labels_file`, `augment`).
Since Python dataclasses are mutable references, the train config mutations carry over when
the same object is reused for the val config. It works by accident (val overwrites train's
values), but is fragile.

**Fix**: Use `dataclasses.replace(video_config, labels_file=..., augment=...)` to create
independent copies for train and val.

### 0.6 `squeeze()` on Batch-Size-1 Breaks Metrics

**Files**: `training/video/train.py:594,828` and `training/multimodal/train.py`

```python
logits = model(frames).squeeze()
```

If the last batch has exactly 1 sample, `squeeze()` removes all dimensions, turning `(1,)`
into a scalar `()`. Downstream `preds == labels` broadcasts unexpectedly.

**Fix**: Use `squeeze(-1)` or `squeeze(1)` to only remove the specific dimension.

### 0.7 GPU Memory Accumulation During Metrics

**Files**: Both `train.py` files accumulate all logits/labels per epoch on GPU:

```python
all_logits.append(logits.detach())  # stays on GPU
```

For large datasets this consumes significant memory. Fix: add `.cpu()`.

### 0.8 Temporal Jitter Bias

**File**: `src/synchronai/data/video/dataset.py:271-276`

The temporal offset is drawn from `[-max_offset, max_offset]` then clamped to `[0, ...)`.
50% of the time the offset is zeroed, producing a biased augmentation. Also, no upper
bound check ensures `second + offset + window_seconds` stays within video duration.

**Fix**: Sample from `[0, max_offset]` directly and add upper bound check.

### 0.9 Subject ID Leakage -- DONE

**Status**: Fixed. Subject-level grouping is enforced across all datasets via
`group_by="subject_id"` default. The multimodal `create_multimodal_splits()` splits
at the group level using set operations.

### 0.10 Deprecated `torch.cuda.amp` API -- DONE

**Status**: Fixed. All training loops now use `from torch.amp import GradScaler, autocast`
with `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`.

---

## Phase 1: Person-Aware Processing (Highest Impact) -- PARTIAL

**Status**: Person-aware classifier and cross-person attention modules implemented as
opt-in alternatives (`person_aware_classifier.py`, `cross_person_attention.py`). Not yet
integrated into standard training pipeline. Person detection (upstream YOLO bounding boxes)
assumed to be handled externally.

**Goal**: Make the model explicitly represent each person in the dyad separately, then
model their coupling. This is the single most impactful architectural change because
synchrony is fundamentally a two-person phenomenon, but the current model sees one
undifferentiated scene.

### 1.1 Add Person Detection + Cropping

Use the existing YOLO26s model (which is already a detector) to produce person bounding
boxes, then crop each person separately.

**New file**: `src/synchronai/models/cv/person_extractor.py`

```
Pipeline:
  Full frame → YOLO26s detect → 2 bounding boxes (adult, child)
  → Crop + resize each to 224x224
  → Feed each through shared feature encoder
```

Assignment heuristic: larger bounding box = adult, smaller = child (robust for
parent-child dyads where the size difference is large).

**Caveat**: This heuristic may fail when the child is standing and the parent is
sitting/crouching. Add a confidence threshold and allow manual verification of a
random subset of assignments.

### 1.2 Add Skeleton Extraction (RTMPose)

Extract pose keypoints per person per frame. This provides structured, interpretable
features that directly encode body movement.

**Recommended**: RTMPose (part of MMPose)
- 90+ FPS on GPU, good accuracy
- 17 COCO keypoints per person per frame
- Install: `pip install mmpose mmdet`

**Critical caveat -- child pose accuracy**: RTMPose/ViTPose have **15-30% accuracy
drops on children** vs adults because training data (COCO, MPII) is overwhelmingly
adult bodies. Child body proportions differ (larger head-to-body ratio, shorter limbs).
Occlusion is severe in close parent-child interactions (child on parent's lap,
parent's arms wrapping around child).

**Required mitigations**:
- Fine-tune pose model on domain-specific data if possible (even a few hundred
  annotated frames helps significantly)
- Add temporal smoothing (Kalman filter or exponential moving average) to reduce
  frame-to-frame jitter in noisy keypoints
- Use identity tracking (SORT/DeepSORT) to prevent person ID swaps when bodies overlap
- Set minimum confidence threshold on keypoints; mask low-confidence joints rather
  than using noisy estimates

**Pre-extraction**: Run offline, save as `.npy` files alongside videos to avoid
runtime overhead.

### 1.3 Cross-Person Temporal Model

**New file**: `src/synchronai/models/cv/dyadic_model.py`

```python
class DyadicSynchronyModel(nn.Module):
    """
    Process each person through shared encoder, then model coupling.

    Architecture:
      Person A frames → Shared backbone → (batch, T, D) features_a
      Person B frames → Shared backbone → (batch, T, D) features_b
      Cross-person attention: features_a attends to features_b (and vice versa)
      Temporal aggregation → synchrony prediction
    """
    def __init__(self, backbone, hidden_dim=256, n_heads=4):
        self.backbone = backbone  # Shared encoder (YOLO, DINOv2, etc.)
        self.cross_person_attn = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.temporal_pool = TemporalLSTM(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, frames_a, frames_b):
        # frames_a, frames_b: (batch, T, C, H, W) -- per-person crops
        feat_a = self.backbone(frames_a)  # (batch, T, D)
        feat_b = self.backbone(frames_b)  # (batch, T, D)

        # Cross-person attention (weight-tied: A→B shares weights with B→A)
        coupled_a = self.cross_person_attn(feat_a, feat_b)  # A queries B
        coupled_b = self.cross_person_attn(feat_b, feat_a)  # B queries A

        # Combine and aggregate
        coupled = coupled_a + coupled_b  # symmetric combination
        pooled = self.temporal_pool(coupled)
        return self.head(pooled)
```

**Design constraints for small data**:
- Use **1-2 cross-attention layers** max (not deeper)
- **Weight-tie** the two directions (since synchrony is symmetric)
- Add dropout (0.1-0.3) on attention weights
- Consider attention entropy regularization to prevent degenerate solutions
  (ignoring one person entirely)

### 1.4 Skeleton + Graph Network Path

For the skeleton-based approach, use a two-person spatial-temporal graph:

```
Graph nodes: 17 joints x 2 people = 34 nodes per frame
Spatial edges: skeleton bones (intra-person) + inter-person edges
  (e.g., A's right hand ↔ B's right hand, A's head ↔ B's head)
Temporal edges: same joint across consecutive frames
```

**Important**: Use **CTR-GCN** (learns channel-specific graph topologies) or **InfoGCN**
(information-bottleneck, more robust to noise), NOT the original ST-GCN which amplifies
noisy keypoints through graph convolutions. This is critical given the expected child
pose estimation noise. Available via `pyskl` toolbox.

### 1.5 Data Pipeline Changes

The current `VideoWindowDataset` returns full frames. Extend it to also return per-person
crops and/or skeleton sequences.

**New file**: `src/synchronai/data/video/person_dataset.py`

**Recommended approach**: Offline preprocessing
- Script: `scripts/preprocess_persons.py`
- Input: video files + labels CSV
- Output: per-second person crops (2 images per frame) + skeleton keypoints (`.npy`)
- Run once, use for all subsequent training

---

## Phase 2: Fix Multimodal Fusion Architecture -- DONE

**Status**: Completed (2026-03-23). See Phase 0.1 for details. Cross-attention now operates
on temporal sequences with residual connections and LayerNorm. The fusion model automatically
routes frame-level features to cross-attention and pooled features to concat/gated.

**Goal**: Replace the degenerate single-token attention with real temporal cross-modal fusion.

### 2.1 Temporal Token Fusion

The key insight: fusion should happen *before* temporal pooling, on the full sequence of
per-frame/per-timestep features.

```
Current (broken):
  Video: (B, T, D) → LSTM → (B, D) → unsqueeze → (B, 1, D) → attention [no-op]

Fixed:
  Video: (B, T_v, D) -- keep full temporal sequence
  Audio: (B, T_a, D) -- keep full temporal sequence
  Cross-attention between T_v video tokens and T_a audio tokens
  Then pool temporally → (B, D) → head
```

**Modified file**: `src/synchronai/models/multimodal/fusion_modules.py`

Replace `CrossModalAttention` with `TemporalCrossModalAttention`:

```python
class TemporalCrossModalAttention(nn.Module):
    """Cross-modal attention over temporal token sequences.

    Uses simple nn.TransformerDecoderLayer (NOT Perceiver IO or MBT).
    With ~59K labeled samples, simpler architectures with stronger
    inductive biases outperform general-purpose ones.
    """

    def __init__(self, video_dim, audio_dim, hidden_dim, num_heads=4, num_layers=2):
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Learnable temporal position encodings
        self.video_pos = nn.Parameter(torch.randn(1, 24, hidden_dim) * 0.02)
        self.audio_pos = nn.Parameter(torch.randn(1, 50, hidden_dim) * 0.02)

        # Cross-modal transformer layers (2-3 layers max for this data size)
        self.cross_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # Temporal pooling after fusion
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, video_tokens, audio_tokens):
        # video_tokens: (B, T_v, video_dim) -- per-frame features
        # audio_tokens: (B, T_a, audio_dim) -- per-frame WavLM/Whisper features
        v = self.video_proj(video_tokens) + self.video_pos[:, :video_tokens.size(1)]
        a = self.audio_proj(audio_tokens) + self.audio_pos[:, :audio_tokens.size(1)]

        for layer in self.cross_layers:
            v = layer(v, a)  # video queries attend to audio keys

        return self.pool(v.transpose(1, 2)).squeeze(-1)  # (B, hidden_dim)
```

### 2.2 Propagate Frame-Level Features Through the Pipeline

This requires changes to how features flow through the fusion model:

**Modified file**: `src/synchronai/models/multimodal/fusion_model.py`

Currently `forward()` calls:
```python
video_output = self.video_model(video_frames, return_features=True)
video_features = video_output['temporal_features']  # Already pooled to (B, D)
```

Change to use `frame_features` (pre-pooling):
```python
video_features = video_output['frame_features']  # (B, T, D) -- per-frame
```

For audio, the encoder currently pools and returns `(B, D)`.
Return the full sequence of content frames instead using `return_sequence=True`.

### 2.3 Three-Way Fusion (Video + Audio + fNIRS)

For three-modality fusion, use a **simple bottleneck approach** (NOT Perceiver IO --
it overfits with <50K samples and requires complex configuration):

```python
class BottleneckFusion(nn.Module):
    """
    Lightweight bottleneck tokens shuttle information between modality streams.
    Handles heterogeneous modalities with different temporal resolutions.
    """
    def __init__(self, hidden_dim=256, n_bottleneck=8, n_heads=4, n_layers=2):
        self.bottleneck = nn.Parameter(torch.randn(1, n_bottleneck, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            BottleneckLayer(hidden_dim, n_heads)
            for _ in range(n_layers)  # 2 layers, not 4+
        ])

    def forward(self, video_tokens, audio_tokens, fnirs_tokens=None):
        b = self.bottleneck.expand(video_tokens.size(0), -1, -1)
        for layer in self.layers:
            b = layer(b, video_tokens, audio_tokens, fnirs_tokens)
        return b.mean(dim=1)  # (B, hidden_dim)
```

This naturally handles different temporal resolutions (video at 12Hz, audio at ~50Hz,
fNIRS at ~8Hz) and missing modalities (just don't pass that input).

---

## Phase 3: Audio Backbone Upgrade -- WavLM-base-plus (BENCHMARKED)

**Status**: WavLM-base-plus (768-dim, 12 layers) implemented and benchmarked.
Feature extraction pipeline operational (`scripts/extract_audio_features.py`).

### Sweep v1 Results (2026-03-23)

Pre-extracted WavLM-base-plus features with equal-weight layer blending, 8 head variants:

| Run | Best AUC | Best Epoch | Notes |
|-----|----------|------------|-------|
| attention | 0.6787 | 1 | All peaked at epoch 1 |
| label_smooth | 0.6775 | 1 | |
| mixup | 0.6775 | 1 | |
| baseline | 0.6773 | 1 | |
| heavy_reg | 0.6770 | 1 | |
| small_cap | 0.6755 | 1 | |
| lstm | 0.6706 | 2 | |

**Diagnosis**: All models peak at epoch 1, val accuracy stuck at ~0.627 (near majority
class ratio). The equal-weight blended 768-dim features lack discriminative signal and
heads overfit immediately.

### Sweep v2 (Submitted 2026-03-24, In Progress)

Two improvements:
1. **Per-layer extraction**: Save all 12 hidden states separately `(13, 50, 768)`.
   Learnable `layer_weights` in the classifier discover which layers matter.
2. **Projection bottleneck**: Compress 768→32/64/128/256 before temporal aggregation,
   forcing compact representations.

10 training jobs: 5 per-layer variants + 5 blended-with-projection variants.
Submit: `sh scripts/bsub/pre_wavlm_audio_sweep_v2_bsub.sh`

### Key Implementation Files

- Encoder: `src/synchronai/models/audio/wavlm_encoder.py` (frozen, with `extract_all_layers()`)
- Extraction: `scripts/extract_audio_features.py` (`--save-all-layers` flag)
- Feature dataset: `src/synchronai/data/audio/feature_dataset.py`
- Training: `scripts/train_audio_from_features.py` (`--project-dim` flag)
- Cluster: `scripts/bsub/pre_wavlm_audio_sweep_v2_bsub.sh`

### 3.2 Speaker Diarization

For dyadic synchrony, knowing *who* is speaking at each moment is highly informative.

**Since we have single-microphone audio**, `pyannote.audio` is the main option, but it has
**40-60% DER (Diarization Error Rate)** out-of-the-box on parent-child conversations:
- Child vocalizations (babbling, crying) confuse VAD
- High overlap rates in parent-child speech degrade clustering
- Fine-tuning on your own data is **mandatory**

**Required steps if using pyannote**:
1. Fine-tune on a subset of your data with manual speaker annotations
2. Consider the Voice Type Classification model from Lavechin et al. (2020) as a
   preprocessing step designed specifically for child-adult speech differentiation
3. Validate DER on held-out data before integrating

**Alternative**: Use WavLM features to implicitly learn speaker information instead
of explicit diarization. The utterance mixing pre-training gives WavLM strong speaker
separation abilities that the downstream model can leverage.

**Recommendation**: Start without explicit diarization. WavLM's utterance mixing
training provides implicit speaker awareness. Add pyannote later if ablation shows
explicit speaker labels help, and invest in fine-tuning it.

---

## Phase 4: Video Backbone Upgrade -- DONE

**Status**: DINOv2 implemented as primary backbone (2026-03). Default switched to
`dinov2-small` based on hyperparameter sweep results (best: `small_heavy_reg`,
AUC 0.697, dropout=0.7, wd=0.01). DINOv2-base overfits more (AUC 0.66). YOLO remains
available as legacy option. Progressive resolution training implemented in
`scripts/train_progressive_features.py`. Multi-resolution stage 1 currently running
(2026-03-26).

**Goal**: Replace YOLO26s feature extraction with a model that provides richer visual
representations for synchrony.

### 4.1 DINOv2-large (Primary Recommendation)

DINOv2 provides excellent per-frame features with practical resource requirements:
- ~300M params, ~3-4 GB VRAM
- Strong semantic features learned via self-supervised learning
- Patch features enable semantic correspondence between body parts of person A
  and person B without explicit pose estimation

```python
from transformers import AutoModel
dino = AutoModel.from_pretrained("facebook/dinov2-large")
# Extract per-frame features, then use existing LSTM for temporal aggregation
```

Use with your existing LSTM temporal aggregation. This is a drop-in backbone
replacement for YOLO26s.

### 4.2 VideoMAE-base (Alternative)

If temporal dynamics need to be captured natively in the backbone:
- VideoMAE-base (~86M params, tractable on single GPU)
- Self-supervised via masked video reconstruction
- Available on HuggingFace: `MCG-NJU/videomae-base-finetuned-kinetics`
- Produces temporal token sequences natively

### 4.3 V-JEPA: AVOID

V-JEPA ViT-H requires **18-24 GB VRAM per sample** for inference. The
`facebookresearch/jepa` codebase is research code designed for Meta's multi-A100 clusters:
- No official fine-tuning script provided
- Default configs assume multi-node training with 8+ A100s
- Multiple GitHub issues report OOM at batch size 1 on single GPUs
- Temporal resolution is coarsened by 2x (16 frames become 8 temporal tokens)
- ~200-400ms per clip on A100 -- not real-time

### 4.4 Backbone Registry

To support swapping backbones easily, add a registry pattern:

**Modified file**: `src/synchronai/models/cv/YOLO_classifier.py`

```python
BACKBONE_REGISTRY = {
    "yolo26s": YOLOFeatureExtractor,
    "dinov2": DINOv2Encoder,
    "videomae": VideoMAEEncoder,
}

def create_backbone(name, **kwargs):
    return BACKBONE_REGISTRY[name](**kwargs)
```

---

## Phase 5: fNIRS Integration -- PLAN READY

**Status**: fNIRS DDPM v3 converged (epoch 621, best val loss 0.003452 at epoch 491).
Detailed transfer learning plan in `docs/plans/fnirs_transfer_plan.md`.

**Summary**: Three-stage transfer learning pipeline:
1. **Stage A**: Train child vs adult classifier on U-Net encoder features (sweep)
2. **Stage B**: Transfer learned representations to synchrony classification
3. **Stage C**: Three-way fusion (video + audio + fNIRS) joined by subject_id

**Architecture**: Each modality has an independent extraction pipeline producing
`feature_index.csv` files keyed by `(subject_id, second)`. Fusion joins on
subject_id at training time — no cross-dependencies during extraction.

**Key dimensions**: U-Net depth=3 downsamples 472 timesteps (60s @ 7.8Hz) to
59 bottleneck timesteps (~1 per second). Bottleneck dim=512, multiscale=960.

**Current status**: Phases 1-2 implemented. Child/adult sweep ready to run.

See `docs/plans/fnirs_transfer_plan.md` for full phased implementation plan.

---

## Phase 6: Progressive Growing & Curriculum Training -- IN PROGRESS

**Status**: Progressive resolution training script implemented
(`scripts/train_progressive_features.py`). Multi-resolution stage 1 currently running
(2026-03-26). DINOv2 hyperparameter sweep completed — best: `small_heavy_reg` (AUC 0.697,
dropout=0.7, wd=0.01). DINOv2-small confirmed as optimal backbone (smaller generalizes
better than base on this dataset size).

**Goal**: Combat overfitting and improve convergence by progressively increasing model
complexity, input resolution, and data difficulty. The DINOv2 sweep (small backbone)
shows clear overfitting — train loss steadily decreases (0.59 → 0.46) while val loss
plateaus at ~0.60 by epoch 8 and diverges after. Progressive growing addresses this
by ensuring each stage of model complexity is fully supported by the data before
adding more capacity.

### Motivation from Current Results

DINOv2-small sweep (`runs/dinov2_sweep/small_backbone/history.json`):
- Best val AUC: 0.692 (epoch 22)
- Best val loss: 0.609 (epoch 22) — but val loss was already oscillating by epoch 15
- Train accuracy reaches 81% while val accuracy stalls at 67%
- ~14% train/val accuracy gap = significant overfitting

Progressive growing provides implicit regularization by limiting model capacity early,
letting the model learn robust features before gaining access to more parameters.

### 6.1 Progressive Resolution Training (Video)

Start training at low resolution and increase as training progresses. DINOv2 ViTs
handle this natively — position embeddings are interpolated to the new resolution,
and all other weights transfer directly.

```python
RESOLUTION_SCHEDULE = [
    {"resolution": 112, "epochs": 10, "lr": 3e-4},   # Fast, coarse features
    {"resolution": 168, "epochs": 10, "lr": 1e-4},   # Intermediate
    {"resolution": 224, "epochs": 15, "lr": 3e-5},   # Full resolution, fine details
]

def progressive_resolution_train(model, dataset, schedule):
    """Train with progressively increasing input resolution.

    Each resolution stage warm-starts from the previous.
    Lower resolutions act as regularization: the model must learn
    features that survive downsampling before seeing fine details.
    """
    for stage in schedule:
        res = stage["resolution"]
        dataset.set_transform(
            transforms.Compose([
                transforms.Resize((res, res)),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        )
        # Rebuild dataloader with new transforms
        loader = DataLoader(dataset, batch_size=adjust_batch_for_res(res), ...)

        # Interpolate DINOv2 position embeddings to new resolution
        if hasattr(model.backbone, 'interpolate_pos_encoding'):
            model.backbone.interpolate_pos_encoding(res)

        optimizer = build_optimizer(model, lr=stage["lr"])
        train_epochs(model, loader, optimizer, n_epochs=stage["epochs"])
```

**Why this helps**: At 112×112, the model sees 4x fewer pixels per frame.
It must learn coarse spatial features (body position, proximity, gross movement)
before fine-grained details (facial expression, hand gestures). This is a natural
regularizer — the model can't overfit to pixel-level noise at low resolution.

**Batch size scaling**: Lower resolution = smaller tensors = larger batch sizes.
At 112×112 you can likely fit 2-4x the batch size, which also stabilizes gradients
in early training.

**Note on YOLO preprocessing**: If using YOLO26s backbone instead of DINOv2,
progressive resolution still works but uses letterbox resizing with pad=114,
not ImageNet-style resize+normalize.

### 6.2 Progressive Unfreezing (Granular Layer-Wise)

Replace the binary freeze/unfreeze (Stage 1 → Stage 2) with gradual unfreezing
from the head backward through the backbone, each group getting a lower learning rate.

```python
# For DINOv2-small: 12 transformer blocks (0-11)
UNFREEZE_SCHEDULE = [
    # (epochs, layers_to_unfreeze, backbone_lr_multiplier)
    (8,  [],          0.0),    # Stage 1: head only
    (6,  [10, 11],    0.1),    # Last 2 blocks: most task-specific
    (6,  [8, 9],      0.05),   # Next 2 blocks
    (6,  [4, 5, 6, 7], 0.02),  # Middle blocks
    (6,  [0, 1, 2, 3], 0.01),  # Early blocks: most general features
]

def progressive_unfreeze_train(model, train_loader, val_loader, schedule):
    """Progressively unfreeze backbone layers from top to bottom.

    Rationale (ULMFiT, Howard & Ruder 2018):
    - Later layers learn task-specific features → unfreeze first
    - Earlier layers learn general features (edges, textures) → keep frozen longer
    - Each newly unfrozen group gets a smaller LR to preserve pre-trained features
    - Prevents catastrophic forgetting of pre-trained representations
    """
    head_lr = 3e-4
    unfrozen_blocks = []

    for n_epochs, new_blocks, lr_mult in schedule:
        unfrozen_blocks.extend(new_blocks)

        # Freeze everything, then selectively unfreeze
        for param in model.backbone.parameters():
            param.requires_grad = False

        for block_idx in unfrozen_blocks:
            for param in model.backbone.blocks[block_idx].parameters():
                param.requires_grad = True

        # Build parameter groups with differential LRs
        param_groups = []
        for block_idx in unfrozen_blocks:
            block_lr = head_lr * lr_mult * (0.9 ** (max(unfrozen_blocks) - block_idx))
            param_groups.append({
                "params": [p for p in model.backbone.blocks[block_idx].parameters()
                          if p.requires_grad],
                "lr": block_lr,
            })
        # Head always trains at full LR
        param_groups.append({
            "params": [p for p in model.head.parameters() if p.requires_grad],
            "lr": head_lr,
        })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs, T_mult=1)
        train_epochs(model, train_loader, optimizer, scheduler, n_epochs)

        # Decay head_lr for next stage
        head_lr *= 0.7
```

**Layer grouping for other backbones**:
- **YOLO26s**: `model.model.model[:10]` — group layers [8,9], [6,7], [4,5], [0-3]
- **WavLM-large**: 24 transformer layers — group in sets of 4-6
- **Whisper large-v3**: 32 encoder layers — group in sets of 8

**Key constraint**: Always filter to `requires_grad` only when building optimizer
groups (existing pattern from memory — never track frozen params in optimizer).

### 6.3 Progressive Modality Fusion

Train modalities independently first, then progressively combine them.
This prevents the degenerate fusion problem (Phase 0.1) by ensuring each
modality stream produces useful features before the fusion layer sees them.

```
Phase A: Video-only → train to convergence (est. val AUC ~0.69)
Phase B: Audio-only → train to convergence (establish audio baseline)
Phase C: Freeze both encoders, train fusion layers only
          (fusion learns to combine already-useful features)
Phase D: Unfreeze all, joint fine-tuning with reduced LR
Phase E: Add fNIRS stream the same way (freeze V+A, train fNIRS+fusion)
```

```python
class ProgressiveMultimodalTrainer:
    """Orchestrates progressive modality fusion training.

    Each phase warm-starts from the previous, so no training is wasted.
    The fusion layer can't collapse to ignoring one modality because
    both streams are already producing useful gradients by Phase C.
    """

    def train_phase_a(self):
        """Video-only training."""
        self.video_model = train_video_classifier(self.video_config)
        save_checkpoint(self.video_model, "phase_a_video.pt")

    def train_phase_b(self):
        """Audio-only training."""
        self.audio_model = train_audio_classifier(self.audio_config)
        save_checkpoint(self.audio_model, "phase_b_audio.pt")

    def train_phase_c(self):
        """Fusion training with frozen encoders."""
        self.multimodal = MultimodalModel(
            video_encoder=load_checkpoint("phase_a_video.pt"),
            audio_encoder=load_checkpoint("phase_b_audio.pt"),
        )
        # Freeze both encoders
        self.multimodal.video_encoder.freeze()
        self.multimodal.audio_encoder.freeze()
        # Only fusion layers and final head are trainable
        train_multimodal(self.multimodal, lr=1e-3, epochs=15)
        save_checkpoint(self.multimodal, "phase_c_fusion.pt")

    def train_phase_d(self):
        """Joint fine-tuning with progressive unfreezing."""
        self.multimodal = load_checkpoint("phase_c_fusion.pt")
        # Progressive unfreeze both encoders (6.2 schedule)
        progressive_unfreeze_train(self.multimodal, ...)

    def train_phase_e(self):
        """Add fNIRS (only if fNIRS-only baseline > chance, see 5.5)."""
        self.multimodal = load_checkpoint("phase_d_joint.pt")
        self.multimodal.add_fnirs_branch(fnirs_encoder)
        # Freeze V+A, train fNIRS branch + updated fusion
        self.multimodal.freeze_video()
        self.multimodal.freeze_audio()
        train_multimodal(self.multimodal, lr=1e-3, epochs=10)
        # Then unfreeze all for final joint fine-tuning
```

**Validation protocol**: At each phase transition, log val metrics to confirm
performance hasn't regressed. If Phase C (fusion) performs worse than Phase A
(video-only), the fusion architecture needs debugging before proceeding.

### 6.4 Progressive Temporal Context

Start with short temporal windows and progressively extend to capture longer-range
synchrony patterns. This maps directly onto the Multi-Scale TCN (Phase 7):

```python
TEMPORAL_SCHEDULE = [
    {"window_sec": 3,   "tcn_levels": 2, "epochs": 10},  # Micro-synchrony
    {"window_sec": 10,  "tcn_levels": 4, "epochs": 10},  # Meso-synchrony
    {"window_sec": 30,  "tcn_levels": 5, "epochs": 8},   # Meso → macro
    {"window_sec": 60,  "tcn_levels": 6, "epochs": 8},   # Macro-synchrony
]

def progressive_temporal_train(model, base_dataset, schedule):
    """Progressively increase temporal context and TCN depth.

    New TCN levels are initialized near-identity so existing
    representations aren't destroyed when the network grows.

    Aligns with Feldman's synchrony hierarchy:
      - Micro (0.1-3s): behavioral mirroring, gaze following
      - Meso (3-30s): turn-taking, shared attention
      - Macro (30s+): overall interaction quality
    """
    for stage in schedule:
        # Extend dataset window
        base_dataset.set_window_seconds(stage["window_sec"])

        # Add new TCN levels (if needed) with near-identity init
        while len(model.tcn.levels) < stage["tcn_levels"]:
            new_level = create_tcn_level(
                model.tcn.hidden_dim,
                dilation=2 ** len(model.tcn.levels)
            )
            # Initialize near-identity: output ≈ input
            init_residual_near_identity(new_level)
            model.tcn.levels.append(new_level)
            model.tcn.scale_heads.append(nn.Linear(model.tcn.hidden_dim, 1))

        optimizer = build_optimizer(model, lr=1e-4)
        train_epochs(model, DataLoader(base_dataset, ...), optimizer,
                     n_epochs=stage["epochs"])
```

**Near-identity initialization**: New TCN layers use residual connections where
the skip path weight starts at ~1.0 and the new convolution path starts near 0.
This ensures adding a layer doesn't disrupt learned representations:

```python
def init_residual_near_identity(layer):
    """Initialize a residual block so output ≈ input (identity shortcut dominates)."""
    nn.init.zeros_(layer[-2].weight)  # Conv weight → 0
    nn.init.zeros_(layer[-2].bias)    # Conv bias → 0
    # The residual connection carries the signal until the layer learns something useful
```

### 6.5 Curriculum Learning by Annotator Agreement

Use inter-rater reliability (Phase 8) as a difficulty proxy. Train on high-agreement
"easy" samples first, then progressively introduce ambiguous cases.

```python
def build_curriculum_sampler(dataset, irr_scores, n_stages=3):
    """Create a curriculum sampler that progressively includes harder samples.

    Args:
        dataset: Full training dataset
        irr_scores: Dict mapping (video_path, second) → annotator agreement rate [0, 1]
        n_stages: Number of curriculum stages

    Returns:
        List of SubsetRandomSamplers, one per stage
    """
    # Sort samples by difficulty (agreement rate)
    difficulties = []
    for idx in range(len(dataset)):
        key = (dataset.samples[idx]['video_path'], dataset.samples[idx]['second'])
        agreement = irr_scores.get(key, 0.5)  # Default to medium if unknown
        difficulties.append((idx, agreement))

    difficulties.sort(key=lambda x: -x[1])  # High agreement (easy) first

    # Stage boundaries
    samplers = []
    for stage in range(1, n_stages + 1):
        cutoff = int(len(difficulties) * stage / n_stages)
        indices = [d[0] for d in difficulties[:cutoff]]
        samplers.append(SubsetRandomSampler(indices))

    return samplers

# Usage in training loop:
CURRICULUM_SCHEDULE = [
    {"fraction": 0.5, "epochs": 8},   # Top 50% agreement (easy)
    {"fraction": 0.8, "epochs": 6},   # Top 80% (medium)
    {"fraction": 1.0, "epochs": 10},  # Full dataset (all difficulties)
]

for stage in CURRICULUM_SCHEDULE:
    sampler = curriculum_samplers[stage_idx]
    loader = DataLoader(dataset, sampler=sampler, ...)
    train_epochs(model, loader, optimizer, n_epochs=stage["epochs"])
```

**Prerequisite**: Requires IRR analysis (Phase 8) to be completed first.
If IRR data is not yet available, a simpler proxy is prediction confidence
from an initial baseline model — samples the model is confident about are
likely "easier."

**Anti-curriculum variant**: Some work (Bengio et al., 2009) shows that
anti-curriculum (hard examples first) can work for certain tasks. Worth an
ablation, but standard curriculum is the safer default.

### 6.6 Combined Progressive Schedule

These techniques compose. A full progressive training run would look like:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Single modality, low resolution, frozen backbone       │
│   - Video at 112×112, head-only training                        │
│   - Curriculum: easy samples only (top 50% agreement)           │
│   - Window: 3 seconds (micro-synchrony)                        │
│   - ~10 epochs                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: Higher resolution, begin unfreezing                    │
│   - Video at 224×224, unfreeze last 2 backbone blocks           │
│   - Curriculum: top 80% agreement                               │
│   - Window: 3 seconds                                           │
│   - ~8 epochs                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Stage 3: Add audio modality                                     │
│   - Freeze video encoder, train audio + fusion                  │
│   - Full curriculum (all samples)                               │
│   - Window: 10 seconds (meso-synchrony)                         │
│   - ~10 epochs                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Stage 4: Joint fine-tuning                                      │
│   - Progressive unfreeze both encoders                          │
│   - Window: 30 seconds                                          │
│   - TCN levels 1-5 active                                       │
│   - ~15 epochs                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Stage 5: Add fNIRS (if baseline > chance)                       │
│   - Freeze V+A, train fNIRS + fusion                            │
│   - Window: 60 seconds                                          │
│   - Full TCN (6 levels)                                         │
│   - ~10 epochs                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Total**: ~53 epochs across 5 stages, but each stage is faster than full
training because early stages use lower resolution / fewer samples / frozen
backbones. Estimated wall-clock time is comparable to a single 35-epoch
full-resolution run.

### 6.7 Sweep Configuration for Progressive Growing

To evaluate progressive growing against the current flat training schedule:

```yaml
# configs/sweeps/progressive_growing_sweep.yaml
sweep_name: progressive_growing_ablation
base_config: configs/train/dinov2_classifier.yaml

experiments:
  # Baseline: current flat training (no progressive growing)
  flat_baseline:
    resolution: 224
    freeze_schedule: "binary"       # Stage 1 frozen → Stage 2 all unfrozen
    curriculum: "none"
    window_sec: 2

  # Progressive resolution only
  prog_resolution:
    resolution_schedule: [112, 168, 224]
    freeze_schedule: "binary"
    curriculum: "none"
    window_sec: 2

  # Progressive unfreezing only
  prog_unfreeze:
    resolution: 224
    freeze_schedule: "gradual_4stage"
    curriculum: "none"
    window_sec: 2

  # Curriculum only
  curriculum_only:
    resolution: 224
    freeze_schedule: "binary"
    curriculum: "agreement_3stage"
    window_sec: 2

  # Progressive resolution + unfreezing (recommended combo)
  prog_res_unfreeze:
    resolution_schedule: [112, 168, 224]
    freeze_schedule: "gradual_4stage"
    curriculum: "none"
    window_sec: 2

  # Full progressive (all techniques)
  full_progressive:
    resolution_schedule: [112, 168, 224]
    freeze_schedule: "gradual_4stage"
    curriculum: "agreement_3stage"
    window_sec: 2

metrics:
  primary: val_auc
  secondary: [val_f1, val_loss, train_val_gap]
  # train_val_gap: track overfitting reduction explicitly
```

**Key ablation question**: Which progressive technique contributes most?
The sweep isolates each technique so you can determine whether the benefit
comes from resolution scheduling, unfreezing schedule, curriculum, or
their combination.

---

## Phase 6b: Self-Supervised Pre-Training (Advanced)

**Goal**: Leverage unlabeled dyadic video to learn synchrony-relevant representations
before fine-tuning on labeled data.

### 6.1 Audio-Visual Temporal Alignment (Proxy Task)

Train the model to predict whether audio and video are temporally aligned:

```
Positive: audio_t matches video_t (same time window)
Negative: audio_t' paired with video_t (different time window, same video)
Loss: Binary cross-entropy
```

This is a free supervisory signal (no labels needed) that teaches the model about
temporal correspondence -- directly relevant to synchrony.

### 6.2 Cross-Person Prediction (Proxy Task)

Train the model to predict person B's behavior from person A's behavior:

```
Input: person A's features at times [t-k, ..., t]
Target: person B's features at time t+1
Loss: L2 or cosine similarity in feature space
```

High prediction accuracy = high synchrony. This is conceptually similar to JEPA's
predictive learning but applied specifically to inter-person dynamics.

### 6.3 Wavelet Coherence Self-Supervised Objective

Use wavelet coherence as a self-supervised training objective:

```python
def wavelet_coherence_loss(features_a, features_b, scales=[1, 2, 4, 8, 16]):
    """
    Train model such that feature similarity tracks wavelet coherence
    between the two persons' motion signals across multiple timescales.
    """
    # Compute ground-truth wavelet coherence from raw signals
    wtc = compute_wavelet_coherence(features_a, features_b, scales)
    # Model predicts coherence from its learned features
    predicted_coherence = model.coherence_head(fused_features)
    return F.mse_loss(predicted_coherence, wtc)
```

This provides per-scale supervision without requiring any manual labels and directly
teaches the model to identify multi-scale coupling.

---

## Phase 7: Multi-Scale Temporal Synchrony

**Goal**: Capture synchrony at multiple timescales, from sub-second micro-synchrony to
minutes-scale behavioral matching.

### Theoretical Foundation

Synchrony operates at distinct timescales (Feldman, 2007):
- **Micro** (0.1-3s): Immediate behavioral mirroring, gaze following, vocal matching
- **Meso** (3-30s): Affect matching, conversational turn-taking, shared attention episodes
- **Macro** (30s-minutes): Overall interaction quality, engagement patterns, rapport

The current 2-second window captures only micro-synchrony. The label granularity
(1-second annotations) fundamentally limits the finest temporal resolution but can
be aggregated upward for coarser scales.

### 7.1 Multi-Scale Temporal Convolutional Network (Primary Recommendation)

A Multi-Scale TCN uses dilated convolutions with exponentially increasing receptive fields
to capture patterns at multiple timescales simultaneously, without the vanishing gradient
problems of very deep RNNs:

```python
class MultiScaleTCN(nn.Module):
    """
    Dilated causal convolutions with receptive fields at multiple scales.

    With kernel_size=3 and dilation=[1, 2, 4, 8, 16, 32, 64]:
      Layer 1: receptive field = 3s   (micro-synchrony)
      Layer 2: receptive field = 7s   (micro → meso transition)
      Layer 3: receptive field = 15s  (meso-synchrony)
      Layer 4: receptive field = 31s  (meso → macro transition)
      Layer 5: receptive field = 63s  (macro-synchrony)
      Layer 6: receptive field = 127s (full interaction patterns)

    Aligns with Feldman's bio-behavioral synchrony hierarchy.
    """
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3, n_levels=6):
        super().__init__()
        self.levels = nn.ModuleList()
        for i in range(n_levels):
            dilation = 2 ** i
            self.levels.append(nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else hidden_dim,
                         hidden_dim, kernel_size,
                         padding=(kernel_size - 1) * dilation,
                         dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ))

        # Scale-specific prediction heads
        self.scale_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_levels)
        ])

    def forward(self, x):
        # x: (B, T, D) -- sequence of per-second features
        x = x.transpose(1, 2)  # (B, D, T)
        scale_features = []
        for level in self.levels:
            x = level(x)
            x = x[:, :, :x.size(2)]  # causal: trim padding
            scale_features.append(x)

        # Predict at each scale
        predictions = {}
        for i, (feat, head) in enumerate(zip(scale_features, self.scale_heads)):
            pooled = feat.mean(dim=-1)  # (B, hidden_dim)
            predictions[f'scale_{i}'] = head(pooled)

        return predictions
```

**Why TCN over LSTM for multi-scale**: TCNs have stable gradients across long sequences
(no vanishing/exploding gradient issue), explicit control over receptive field via
dilation factors, and can be computed in parallel (vs sequential RNN processing).

### 7.2 Hierarchical Multi-Scale Prediction

Predict synchrony at three granularities simultaneously:

| Scale | Window | Label Source | Interpretation |
|-------|--------|-------------|----------------|
| 1s (micro) | Per-second | Direct annotation | Immediate behavioral matching |
| 5s (meso) | 5-second sliding | Majority vote of 5 × 1s labels | Turn-taking, shared attention |
| 30s (macro) | 30-second sliding | Majority vote of 30 × 1s labels | Overall interaction quality |

```python
class HierarchicalPredictor(nn.Module):
    def __init__(self, feature_dim):
        self.micro_head = nn.Linear(feature_dim, 1)   # 1s prediction
        self.meso_pool = nn.AvgPool1d(5)               # 5s aggregation
        self.meso_head = nn.Linear(feature_dim, 1)     # 5s prediction
        self.macro_pool = nn.AvgPool1d(30)              # 30s aggregation
        self.macro_head = nn.Linear(feature_dim, 1)    # 30s prediction

    def compute_multiscale_loss(self, features, labels_1s):
        # Micro: direct 1s prediction
        micro_pred = self.micro_head(features)
        micro_loss = F.binary_cross_entropy_with_logits(micro_pred, labels_1s)

        # Meso: majority-vote 5s labels
        labels_5s = (labels_1s.unfold(1, 5, 5).mean(-1) > 0.5).float()
        meso_feat = self.meso_pool(features.transpose(1,2)).transpose(1,2)
        meso_pred = self.meso_head(meso_feat)
        meso_loss = F.binary_cross_entropy_with_logits(meso_pred, labels_5s)

        # Macro: majority-vote 30s labels
        labels_30s = (labels_1s.unfold(1, 30, 30).mean(-1) > 0.5).float()
        macro_feat = self.macro_pool(features.transpose(1,2)).transpose(1,2)
        macro_pred = self.macro_head(macro_feat)
        macro_loss = F.binary_cross_entropy_with_logits(macro_pred, labels_30s)

        return micro_loss + 0.5 * meso_loss + 0.3 * macro_loss
```

### 7.3 Wavelet Transform Coherence (WTC) Features

Wavelet coherence provides a time-frequency decomposition of the coupling between two
persons' signals. This can be used as:

1. **Input features**: Compute WTC between person A and person B's motion signals (from
   keypoints or optical flow) at multiple scales, then feed as additional channels
2. **Self-supervised objective**: Train the model to predict wavelet coherence from its
   learned features (see Phase 6.3)

```python
import pywt  # PyWavelets

def compute_wavelet_coherence(signal_a, signal_b, wavelet='morl', scales=None):
    """
    Compute wavelet coherence between two motion signals.

    Args:
        signal_a, signal_b: (T,) motion signals from person A, B
        wavelet: Mother wavelet (Morlet is standard for synchrony)
        scales: Scales to analyze (correspond to frequencies/timescales)

    Returns:
        coherence: (n_scales, T) coherence values in [0, 1]
    """
    if scales is None:
        scales = np.arange(1, 128)  # 1s to ~128s timescales

    # Continuous wavelet transform for each person
    cwt_a, freqs = pywt.cwt(signal_a, scales, wavelet)
    cwt_b, _ = pywt.cwt(signal_b, scales, wavelet)

    # Cross-wavelet spectrum
    cross = cwt_a * np.conj(cwt_b)

    # Smoothed coherence (Torrence & Compo, 1998)
    smooth_cross = smooth_in_time_and_scale(cross)
    smooth_aa = smooth_in_time_and_scale(np.abs(cwt_a)**2)
    smooth_bb = smooth_in_time_and_scale(np.abs(cwt_b)**2)

    coherence = np.abs(smooth_cross)**2 / (smooth_aa * smooth_bb + 1e-10)
    return coherence  # (n_scales, T) in [0, 1]
```

### 7.4 Temporal Smoothness Loss

Add a regularization term that encourages smooth predictions across time, preventing
the model from producing noisy frame-by-frame oscillations:

```python
def temporal_smoothness_loss(predictions, alpha=0.1):
    """
    Penalize rapid changes in predictions between adjacent timesteps.
    predictions: (B, T) sequence of prediction logits
    """
    diff = predictions[:, 1:] - predictions[:, :-1]
    return alpha * (diff ** 2).mean()
```

This serves as a soft prior that synchrony states tend to persist for multiple seconds
rather than flipping every second.

### 7.5 Data Requirements for Multi-Scale

Multi-scale analysis requires **longer input sequences** than the current 2-second windows:
- Micro: 2-3s windows (current setup works)
- Meso: 10-30s windows (need to load 10-30 consecutive seconds)
- Macro: 60-120s windows (need minute-scale context)

**Dataset adaptation**:
- Modify `VideoWindowDataset` to support variable-length windows
- For meso/macro, extract features per-second (using existing 2s windows), then
  concatenate feature sequences for longer-range modeling
- The Multi-Scale TCN naturally handles this: feed per-second features as a sequence

---

## Phase 8: Inter-Rater Reliability (IRR) Analysis

**Goal**: Establish a human performance ceiling for the synchrony prediction task.
No model can exceed the agreement level of expert human coders.

### 8.1 Design

The CARE synchrony study labels come from multiple annotators per session, with conflict
resolution tracked in `raw_to_csv.py` via `combine_label_files()`. This provides the
raw data needed for IRR analysis.

**Metrics to compute**:

| Metric | Purpose | Interpretation |
|--------|---------|---------------|
| Cohen's Kappa (κ) | Agreement beyond chance for each annotator pair | κ > 0.8 = excellent, 0.6-0.8 = good, 0.4-0.6 = moderate |
| Fleiss' Kappa | Multi-rater agreement (if >2 annotators) | Same scale as Cohen's |
| % Agreement | Raw agreement rate | Baseline (doesn't account for chance) |
| Prevalence-Adjusted Bias-Adjusted Kappa (PABAK) | Adjusts for class imbalance | Useful given 44/56% split |
| Krippendorff's Alpha | Handles missing annotations, varying raters | Most robust metric |

### 8.2 Implementation

**New file**: `src/synchronai/evaluation/irr_analysis.py`

```python
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np

def compute_irr(annotator_files: list[str]) -> dict:
    """
    Compute inter-rater reliability from multiple annotator label files.

    Args:
        annotator_files: Paths to per-annotator Excel/CSV files

    Returns:
        Dictionary with IRR metrics and per-pair kappa values
    """
    # Load all annotator labels, aligned by (video_path, second)
    annotations = {}
    for f in annotator_files:
        rater_id = extract_rater_id(f)
        annotations[rater_id] = load_annotations(f)

    # Align to common (video_path, second) pairs
    common_keys = set.intersection(*[set(a.keys()) for a in annotations.values()])

    # Pairwise Cohen's Kappa
    raters = list(annotations.keys())
    pairwise_kappa = {}
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            labels_i = [annotations[raters[i]][k] for k in common_keys]
            labels_j = [annotations[raters[j]][k] for k in common_keys]
            kappa = cohen_kappa_score(labels_i, labels_j)
            pairwise_kappa[(raters[i], raters[j])] = kappa

    mean_kappa = np.mean(list(pairwise_kappa.values()))

    return {
        'pairwise_kappa': pairwise_kappa,
        'mean_kappa': mean_kappa,
        'n_common_samples': len(common_keys),
        'n_raters': len(raters),
        'percent_agreement': compute_percent_agreement(annotations, common_keys),
        'conflict_rate': compute_conflict_rate(annotations, common_keys),
    }
```

### 8.3 Applying IRR to Evaluation

Use the IRR analysis on the holdout validation/test set:

1. **Performance ceiling**: If mean κ = 0.75, the model's theoretical maximum κ is ~0.75.
   A model achieving κ = 0.70 is essentially at human-level.

2. **Difficulty stratification**: Compute IRR per-video or per-session. Videos where
   annotators disagree most are the "hardest" cases. Evaluate model separately on
   easy (high IRR) vs hard (low IRR) subsets.

3. **Confident labels for evaluation**: On the holdout set, weight evaluation by
   annotator agreement. Samples where all annotators agree are "gold standard";
   samples with disagreement should be weighted less or analyzed separately.

4. **Error analysis**: Compare model errors against annotator disagreement patterns.
   If the model and one annotator make the same "errors" against the consensus, the
   model may be learning that annotator's bias.

### 8.4 Script

**New file**: `scripts/compute_irr.py`

```
Usage: python scripts/compute_irr.py --annotator-dir /path/to/annotator/files/
                                       --holdout-csv /path/to/test_labels.csv
                                       --output-dir runs/irr_analysis/

Output:
  - irr_report.json: All metrics (kappa values, agreement rates, conflict patterns)
  - irr_by_session.csv: Per-session IRR metrics
  - confusion_matrices/: Per-pair annotator confusion matrices
  - difficulty_stratification.csv: Per-second difficulty scores based on annotator agreement
```

---

## Phase 9: Unit Tests

**Goal**: Establish test coverage before modifying the codebase. The current test files
(`tests/test_data_loading.py`, `test_generation.py`, `test_inference.py`, `test_training.py`)
are all empty placeholders.

### 9.1 Model Forward Pass Smoke Tests

Verify correct output shapes with minimal input:

```python
# test_models.py

def test_video_classifier_forward():
    """Smoke test: video classifier produces correct output shape."""
    config = VideoClassifierConfig(backbone="yolo26s", temporal="lstm", hidden_dim=64)
    model = VideoClassifier(config)
    frames = torch.randn(2, 24, 3, 640, 640)  # (B=2, T=24, C, H, W)
    out = model(frames)
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"

def test_audio_classifier_forward():
    """Smoke test: audio classifier produces correct output shape."""
    config = AudioClassifierConfig(whisper_model_size="tiny", hidden_dim=64)
    model = AudioClassifier(config)
    audio = torch.randn(2, 16000)  # (B=2, 1 second at 16kHz)
    out = model(audio)
    assert out['event_logits'].shape == (2, 7)

def test_wavlm_encoder_forward():
    """Smoke test: WavLM encoder produces correct feature dimensions."""
    encoder = WavLMEncoder(model_name="microsoft/wavlm-base-plus", freeze=True)
    audio = torch.randn(2, 16000)
    pooled = encoder(audio, return_sequence=False)
    assert pooled.shape == (2, 768)  # base-plus dim
    seq = encoder(audio, return_sequence=True)
    assert seq.shape[0] == 2 and seq.shape[2] == 768

def test_multimodal_fusion_forward():
    """Smoke test: fusion model accepts video + audio and produces sync prediction."""
    # Test with random inputs matching expected dims
    video_tokens = torch.randn(2, 24, 512)
    audio_tokens = torch.randn(2, 50, 1024)
    fusion = TemporalCrossModalAttention(512, 1024, 256, num_heads=4, num_layers=2)
    out = fusion(video_tokens, audio_tokens)
    assert out.shape == (2, 256)

def test_multiscale_tcn_forward():
    """Smoke test: Multi-Scale TCN produces predictions at each scale."""
    tcn = MultiScaleTCN(input_dim=256, hidden_dim=128, n_levels=4)
    x = torch.randn(2, 30, 256)  # 30 seconds of features
    preds = tcn(x)
    assert len(preds) == 4
    for k, v in preds.items():
        assert v.shape == (2, 1)
```

### 9.2 Gradient Flow Tests

Ensure gradients reach all trainable parameters (catches stage-2 type bugs):

```python
def test_gradient_flow_video():
    """Verify gradients flow to all trainable params."""
    model = VideoClassifier(config)
    model.unfreeze_backbone()
    frames = torch.randn(1, 24, 3, 640, 640)
    loss = model(frames).sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

def test_gradient_flow_audio_stage2():
    """Verify gradients flow through unfrozen audio encoder (catches _freeze bug)."""
    model = AudioClassifier(config)
    model.unfreeze_encoder()
    audio = torch.randn(1, 16000)
    loss = model(audio)['event_logits'].sum()
    loss.backward()
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in encoder_grads), \
        "No gradients flowing to encoder in stage 2!"

def test_frozen_params_not_in_optimizer():
    """Verify get_parameter_groups only returns requires_grad params."""
    model = AudioClassifier(config)
    model.freeze_encoder()
    groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-4)
    all_params = [p for g in groups for p in g['params']]
    for p in all_params:
        assert p.requires_grad, "Frozen param found in optimizer groups!"
```

### 9.3 Data Pipeline Tests

Verify data loading, label correctness, and augmentation bounds:

```python
def test_dataset_label_range():
    """Verify all labels are 0 or 1 (binary sync/async)."""
    dataset = VideoWindowDataset(labels_file="test_labels.csv", ...)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample['label'] in (0, 1), f"Invalid label {sample['label']} at idx {i}"

def test_dataset_no_subject_leakage():
    """Verify train and val splits have no overlapping subjects."""
    train_ds, val_ds = create_splits(labels_file, group_by="subject_id")
    train_subjects = {s['subject_id'] for s in train_ds}
    val_subjects = {s['subject_id'] for s in val_ds}
    overlap = train_subjects & val_subjects
    assert len(overlap) == 0, f"Subject leakage: {overlap}"

def test_augmentation_preserves_shape():
    """Verify augmentation doesn't change tensor shapes."""
    dataset = VideoWindowDataset(labels_file="test_labels.csv", augment=True)
    sample = dataset[0]
    assert sample['frames'].shape == (24, 3, 640, 640)

def test_multimodal_alignment():
    """Verify video and audio samples at same index have matching metadata."""
    dataset = MultiModalDataset(labels_file="test_labels.csv")
    sample = dataset[0]
    assert 'video_frames' in sample and 'audio_chunk' in sample
    assert sample['video_path'] is not None
    assert sample['second'] >= 0

def test_squeeze_batch_size_one():
    """Regression test for squeeze() on batch-size-1."""
    logits = torch.tensor([[0.5]])  # (1, 1)
    squeezed = logits.squeeze(-1)    # Should be (1,), NOT scalar
    assert squeezed.dim() == 1, f"squeeze reduced to scalar: {squeezed.shape}"
```

### 9.4 Checkpoint Round-Trip Tests

```python
def test_checkpoint_save_load():
    """Verify model can be saved and loaded with identical weights."""
    model = AudioClassifier(config)
    _ = model(torch.randn(1, 16000))  # Trigger lazy loading
    state = model.state_dict()
    torch.save(state, "/tmp/test_ckpt.pt")

    model2 = AudioClassifier(config)
    model2.load_state_dict(torch.load("/tmp/test_ckpt.pt"))

    for k in state:
        assert torch.equal(state[k], model2.state_dict()[k]), f"Mismatch for {k}"
```

### 9.5 Test Infrastructure

```
tests/
  conftest.py           # Shared fixtures (tiny configs, test data)
  test_models.py        # Forward pass smoke tests (9.1)
  test_gradients.py     # Gradient flow tests (9.2)
  test_data_pipeline.py # Data loading and split tests (9.3)
  test_checkpoints.py   # Save/load round-trip tests (9.4)
  test_irr.py           # IRR computation tests (9.5)
  fixtures/
    tiny_labels.csv     # 50-100 rows for testing
    tiny_video.mp4      # Short test video
```

Use `pytest` with markers:
- `@pytest.mark.slow` for tests requiring model downloads (Whisper, WavLM)
- `@pytest.mark.gpu` for tests requiring CUDA
- Default tests should run on CPU in <30s total

---

## Phase 10: Profiling & Benchmarking

**Goal**: Identify actual bottlenecks before upgrading components. If data loading is the
bottleneck (highly likely with `num_workers=0`), a faster backbone won't help.

### 10.1 Training Pipeline Profile

```python
# scripts/profile_training.py
import torch.profiler
import time

def profile_training_step(model, dataloader, device):
    """Profile a single training epoch to find bottlenecks."""

    # 1. Data loading time
    t0 = time.time()
    batch = next(iter(dataloader))
    data_load_time = time.time() - t0

    # 2. Forward pass time
    t0 = time.time()
    with torch.amp.autocast("cuda"):
        output = model(batch['frames'].to(device))
    forward_time = time.time() - t0

    # 3. Backward pass time
    t0 = time.time()
    loss = F.binary_cross_entropy_with_logits(output, batch['label'].to(device))
    loss.backward()
    backward_time = time.time() - t0

    # 4. GPU memory
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        'data_loading_ms': data_load_time * 1000,
        'forward_ms': forward_time * 1000,
        'backward_ms': backward_time * 1000,
        'peak_gpu_memory_gb': peak_memory_gb,
        'data_loading_pct': data_load_time / (data_load_time + forward_time + backward_time) * 100,
    }
```

### 10.2 Key Metrics to Measure

| Component | Metric | Expected Issue |
|-----------|--------|----------------|
| DataLoader | Time per batch | `num_workers=0` = main bottleneck |
| Video decode | Frames/second | cv2.VideoCapture may be slow |
| YOLO backbone | Forward ms/batch | Likely fine (~5-10ms frozen) |
| Whisper/WavLM | Forward ms/batch | Whisper 30s padding wastes 97% compute |
| Fusion | Forward ms/batch | Should be negligible |
| GPU memory | Peak allocated | Determines max batch size |
| Gradient accumulation | Overhead | AMP scaler + unscale + clip |

### 10.3 Quick Wins to Measure

1. **`num_workers` experiment**: Try num_workers=2, 4, 8 -- typically gives 2-5x
   data loading speedup
2. **Pre-extracted features**: If backbone is frozen (stage 1), extract features once
   and save to disk. Training becomes a simple MLP/LSTM over features -- 10-50x faster
3. **WavLM vs Whisper inference**: WavLM processes 1s natively (50 frames) vs Whisper
   padding to 30s (1500 frames). Expected ~20-30x speedup for audio encoding
4. **Mixed precision impact**: Measure AMP speedup (typically 1.5-2x on modern GPUs)

### 10.4 Benchmark Script

**New file**: `scripts/benchmark.py`

```
Usage: python scripts/benchmark.py --config configs/train/multimodal_classifier.yaml
                                    --n-batches 50
                                    --output runs/benchmarks/

Output:
  - timing_breakdown.json: Per-component timing
  - memory_profile.json: GPU memory usage
  - recommendations.txt: Prioritized optimization suggestions
```

---

## Phase 11: Dataset Size Audit

**Goal**: Understand the dataset characteristics to inform architecture choices.

### 11.1 Current Dataset Statistics

These are computed at runtime from the labels CSV (`scripts/data/labels.csv`):

| Statistic | Value |
|-----------|-------|
| Total labeled seconds | 59,250 |
| Class distribution | ~44% async (0) / ~56% sync (1) |
| Subjects | 50001-50581 |
| Sessions | All V0 |
| Annotators | Multiple per session (conflict resolution tracked) |

### 11.2 Audit Script

**New file**: `scripts/audit_dataset.py`

```python
def audit_dataset(labels_file: str) -> dict:
    """Comprehensive dataset audit."""
    df = pd.read_csv(labels_file)

    audit = {
        'total_samples': len(df),
        'unique_subjects': df['subject_id'].nunique(),
        'unique_videos': df['video_path'].nunique(),
        'class_distribution': df['label'].value_counts().to_dict(),
        'class_balance_ratio': df['label'].value_counts().min() / df['label'].value_counts().max(),
        'sessions': df['session'].unique().tolist() if 'session' in df.columns else None,

        # Per-subject statistics
        'samples_per_subject': df.groupby('subject_id').size().describe().to_dict(),

        # Per-video statistics
        'seconds_per_video': df.groupby('video_path')['second'].max().describe().to_dict(),

        # Label distribution per subject (detect annotation bias)
        'sync_rate_per_subject': df.groupby('subject_id')['label'].mean().describe().to_dict(),

        # Missing data
        'null_subject_ids': df['subject_id'].isna().sum(),
        'null_labels': df['label'].isna().sum(),
    }
    return audit
```

### 11.3 Architecture Sizing Guidelines

The dataset size (~59K samples) informs which architectures are viable:

| Dataset Size | Viable Architectures |
|-------------|---------------------|
| <5K | Linear models, small MLPs, frozen backbone + head only |
| 5K-20K | Current architecture (YOLO+LSTM+MLP), simple cross-attention |
| 20K-100K | **Our range**. DINOv2/WavLM with fine-tuning, cross-person attention, Multi-Scale TCN, 2-layer fusion |
| 100K+ | Perceiver IO, deeper transformers, full end-to-end training |

With ~59K samples, we are in a good range for moderate-complexity models. The key
constraint is the number of unique subjects (~500+), which determines the effective
diversity of the training data for generalization.

---

## Implementation Priority & Sequencing

### Wave 1: Bug Fixes + Testing Infrastructure (1-2 days)
- [ ] **0.1** Switch cross-attention fusion to concat until Phase 2 is implemented
- [ ] **0.2** Remove audio event auxiliary task
- [ ] **0.3** Fix Whisper `.to()` AND `_freeze` flag (SHOWSTOPPER)
- [ ] **0.4** Fix lazy loading / `state_dict()` issue
- [ ] **0.5** Fix config mutation in `create_multimodal_splits`
- [ ] **0.6** Fix `squeeze()` on batch-size-1
- [ ] **0.7** Add `.cpu()` to metric accumulation
- [ ] **0.8** Fix temporal jitter bias + bounds check
- [ ] **0.9** Fix subject_id=None leakage
- [ ] **0.10** Update deprecated `torch.cuda.amp` API
- [ ] **9.1-9.4** Write unit tests for all bug fixes (test BEFORE and AFTER)

### Wave 2: Profiling & Dataset Audit (1 day)
- [ ] **10.1-10.4** Profile training pipeline, identify actual bottlenecks
- [ ] **11.1-11.3** Run dataset audit, confirm architecture viability
- [ ] **8.1-8.4** Compute inter-rater reliability (performance ceiling)
- [ ] Fix `num_workers=0` bottleneck based on profiling results

### Wave 3: Progressive Growing Sweep (3-5 days)
- [ ] **6.7** Run progressive growing ablation sweep (resolution, unfreezing, curriculum variants)
- [ ] **6.1** Implement progressive resolution training for DINOv2
- [ ] **6.2** Implement progressive unfreezing (4-stage gradual vs binary)
- [ ] **6.5** Implement curriculum learning by annotator agreement (requires Wave 2 IRR)
- [ ] Analyze sweep results: which progressive technique helps most?
- [ ] **6.6** Assemble best-performing progressive schedule for use in all subsequent waves

### Wave 4: Audio Backbone Upgrade (3-5 days)
- [ ] **3.1** Add WavLM-large encoder (replace Whisper)
- [ ] **3.2** Evaluate implicit speaker awareness vs explicit diarization
- [ ] Update training pipeline for WavLM
- [ ] Write WavLM-specific unit tests

### Wave 5: Person-Aware Processing (1-2 weeks)
- [ ] **1.1** Add person detection + cropping using existing YOLO
- [ ] **1.2** Add skeleton extraction (RTMPose) -- offline preprocessing
- [ ] **1.3** Implement cross-person attention model (1-2 layers, weight-tied)
- [ ] **1.4** Implement CTR-GCN or InfoGCN for skeleton path (NOT ST-GCN)
- [ ] **1.5** Build person-aware dataset with offline preprocessing

### Wave 6: Proper Fusion (1 week)
- [ ] **2.1** Implement `TemporalCrossModalAttention` with real token sequences
- [ ] **2.2** Propagate frame-level features (skip temporal pooling before fusion)
- [ ] **6.3** Apply progressive modality fusion schedule (train each modality → freeze → fuse)
- [ ] Write fusion-specific tests

### Wave 7: Multi-Scale Temporal Synchrony (1-2 weeks)
- [ ] **7.1** Implement Multi-Scale TCN with dilated convolutions
- [ ] **7.2** Implement hierarchical multi-scale prediction (1s/5s/30s)
- [ ] **6.4** Apply progressive temporal context (grow window + TCN depth incrementally)
- [ ] **7.3** Add wavelet coherence features (optional)
- [ ] **7.4** Add temporal smoothness loss
- [ ] **7.5** Modify dataset to support variable-length windows

### Wave 8: fNIRS Integration (1-2 weeks)
- [ ] **5.1** Train DDPM to convergence, evaluate generation quality
- [ ] **5.4** Extract per-second DDPM bottleneck features offline (120s → 120 per-second features)
- [ ] **5.5** Build standalone fNIRS synchrony classifier (LSTM over per-second features)
- [ ] **5.5** Evaluate fNIRS-only baseline vs chance / video-only / audio-only
- [ ] **5.6** If fNIRS baseline is above chance, integrate into multimodal fusion (three-way)
- [ ] **5.3** If hyperscanning data available, add inter-brain attention module

### Wave 9: Video Backbone Upgrade (1 week, optional)
- [ ] **4.1** Experiment with DINOv2-large (primary) or VideoMAE-base (alternative)
- [ ] **4.4** Add backbone registry for easy swapping

### Wave 10: Self-Supervised Pre-Training (2+ weeks, advanced)
- [ ] **6b.1** Audio-visual temporal alignment proxy task
- [ ] **6b.2** Cross-person prediction proxy task
- [ ] **6b.3** Wavelet coherence self-supervised objective

---

## Key Dependencies

```
New packages needed:
  pip install transformers   # WavLM, DINOv2, VideoMAE (HuggingFace)
  pip install mmpose mmdet   # RTMPose for skeleton extraction
  pip install PyWavelets     # Wavelet coherence features
  pip install pyannote.audio # Speaker diarization (if needed, requires fine-tuning)

Already installed:
  ultralytics               # YOLO26s
  torch, torchvision        # PyTorch
  tensorflow                # fNIRS DDPM
  openai-whisper            # Whisper encoder (being replaced by WavLM)
  scikit-learn              # IRR metrics (Cohen's Kappa)
  pytest                    # Testing
```

---

## Evaluation Strategy

### Metrics

| Metric | Description | Notes |
|--------|-------------|-------|
| Balanced Accuracy | Primary metric (handles class imbalance) | Main comparison metric |
| F1 Score | Harmonic mean of precision/recall | Report for both classes |
| AUC-ROC | Threshold-independent performance | Compare model confidence calibration |
| Cohen's Kappa | Agreement beyond chance | **Compare directly to IRR ceiling** |
| Multi-scale metrics | Kappa at 1s, 5s, 30s windows | For Phase 7 evaluation |

### Performance Ceiling

The inter-rater reliability (Phase 8) sets the achievable ceiling:
- If mean annotator κ = 0.75, model κ = 0.70 is essentially human-level
- Report model performance as % of IRR ceiling (e.g., 70/75 = 93% of ceiling)
- Separate evaluation on "easy" (high IRR) vs "hard" (low IRR) samples

### Ablation Protocol

When adding a new component, compare incrementally:

```
Experiment 0: Baseline (current model, all bugs fixed, flat training)
Experiment 1: + Progressive growing (best schedule from Wave 3 sweep)
Experiment 2: + WavLM (replace Whisper)
Experiment 3: + Person-aware features
Experiment 4: + Cross-person attention
Experiment 5: + Temporal cross-modal fusion (with progressive modality schedule)
Experiment 6: + Multi-scale TCN (with progressive temporal context)
Experiment 7: Full model with all upgrades
```

Each experiment uses identical data splits, seeds, and hyperparameters except
the component being tested.

---

## Approach Summary Table

| Approach | Rating | Key Reason |
|----------|--------|-----------|
| Progressive Growing | **Recommended, high priority** | Addresses 14% train/val gap; progressive resolution + unfreezing + curriculum |
| WavLM-large | **Confirmed, primary** | Utterance mixing, native 1s chunks, paralinguistic features |
| DINOv2-large | **Primary video upgrade** | 300M params, 3-4 GB VRAM, practical V-JEPA replacement |
| Cross-Person Attention | **Confirmed, keep simple** | 1-2 layers, weight-tied, dropout on attention |
| CTR-GCN / InfoGCN | **Use instead of ST-GCN** | More robust to noisy child keypoints |
| Multi-Scale TCN | **Recommended for temporal** | Dilated convolutions, explicit multi-scale receptive fields |
| Simple Cross-Attention Fusion | **Use instead of Perceiver IO** | Better for 59K samples, less overfitting risk |
| DDPM Per-Second Features | **Recommended** | Extract 1s-resolution features from 120s model; standalone classifier first, then fuse |
| VideoMAE-base | **Alternative to DINOv2** | If temporal dynamics need native modeling |
| V-JEPA | **Avoid** | GPU cost, research-only code, coarse temporal resolution |
| Perceiver IO | **Avoid** | Overfits on small data, complex configuration |
| ST-GCN | **Avoid** | Amplifies noisy keypoints |
| pyannote (out-of-box) | **High risk** | 40-60% DER on child speech, fine-tuning mandatory |

---

## Architecture Summary (After All Upgrades)

```
                    +-----------+
                    | Full Frame|
                    +-----+-----+
                          |
                    YOLO26s detect
                          |
              +-----------+-----------+
              |                       |
        Person A crop           Person B crop
              |                       |
        Shared Encoder          Shared Encoder
       (DINOv2 / YOLO)        (DINOv2 / YOLO)
              |                       |
        (B, T, D_v)            (B, T, D_v)
              |                       |
              +----Cross-Person-------+
              |    Attention          |
              +-----------+-----------+
                          |
                    (B, T, D_v)   Video stream
                          |
    +---------------------+---------------------+
    |                                             |
    |    Audio: (B, n_samples)                    |
    |       |                                     |
    |    WavLM-large                              |
    |       |                                     |
    |    Layer-wise weighted sum                  |
    |       |                                     |
    |    (B, T_a, D_a)   Audio stream             |
    |                                             |
    |    fNIRS: 120s window (B, 938, channels)      |
    |       |                                     |
    |    DDPM U-Net bottleneck (t~200)            |
    |       |                                     |
    |    Per-second slicing → (B, 120, D_bneck)   |
    |       |                                     |
    |    (B, T, D_f)   fNIRS stream (1s res)      |
    |                                             |
    +---------------------+-----------------------+
                          |
              Cross-Modal Attention Fusion
              (2-3 layers, NOT Perceiver IO)
                          |
                    (B, D_fused)
                          |
                  Multi-Scale TCN
                          |
              +-----+-----+-----+
              |     |     |     |
           micro  meso  macro  full
            1s    5s    30s    sync
                          |
                   Synchrony Head
                          |
                    sync/async
```

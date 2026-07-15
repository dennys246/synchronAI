# Multi-Modal Transfer Learning Guide

The multi-modal synchrony classifier supports three transfer learning strategies, giving you flexibility in how to leverage pretrained models.

## 🎯 Transfer Learning Strategies

### 1. **Default: Pretrained Backbones (From Scratch)**

**When to use**: First time training, no existing video/audio models

**What happens**:
- ✅ Video: Pretrained YOLO backbone from Ultralytics
- ✅ Audio: Pretrained Whisper encoder from OpenAI
- 🔧 Heads: Trained from scratch on your data

**Configuration**:
```yaml
# Don't specify pretrained paths
pretrained:
  video: null
  audio: null
```

**Shell script**:
```bash
PRETRAINED_VIDEO=""
PRETRAINED_AUDIO=""
```

**Advantages**:
- Simple, no dependencies on prior training
- Strong general-purpose backbones (YOLO for vision, Whisper for audio)
- Good starting point

**Disadvantages**:
- Heads need to learn synchrony/event patterns from scratch
- May take longer to converge

---

### 2. **Load Heads Only (RECOMMENDED)**

**When to use**: You've already trained separate video and audio classifiers

**What happens**:
- ✅ Video: Pretrained YOLO backbone from Ultralytics (kept!)
- ✅ Audio: Pretrained Whisper encoder from OpenAI (kept!)
- 📦 Video head: Loaded from your trained video classifier
- 📦 Audio head: Loaded from your trained audio classifier
- 🔧 Fusion: Trained from scratch

**Configuration**:
```yaml
pretrained:
  video: "runs/video_classifier/best.pt"
  audio: "runs/audio_classifier/best.pt"
  load_heads_only: true  # Key setting!
```

**Shell script**:
```bash
PRETRAINED_VIDEO="runs/video_classifier/best.pt"
PRETRAINED_AUDIO="runs/audio_classifier/best.pt"
LOAD_HEADS_ONLY="true"
```

**Advantages**:
- **Best of both worlds**: Strong pretrained backbones + task-adapted heads
- Faster convergence (heads already know synchrony/events)
- Lower risk of overfitting (backbones stay robust)
- **This is the recommended approach!**

**Training workflow**:
```bash
# Step 1: Train video classifier (once)
bash scripts/bsub/pre_video_synchrony_bsub.sh

# Step 2: Train audio classifier (once)
bash scripts/bsub/pre_audio_synchrony_bsub.sh

# Step 3: Train multi-modal with pretrained heads
# Edit scripts/train_multimodal_synchrony.sh:
PRETRAINED_VIDEO="runs/video_classifier/best.pt"
PRETRAINED_AUDIO="runs/audio_classifier/best.pt"
LOAD_HEADS_ONLY="true"

bash scripts/bsub/pre_multimodal_synchrony_bsub.sh
```

---

### 3. **Load Complete Models**

**When to use**: You want to start exactly from your trained models (rare)

**What happens**:
- 📦 Video: Complete model loaded (backbone + head + temporal)
- 📦 Audio: Complete model loaded (encoder + heads)
- 🔧 Fusion: Trained from scratch

**Configuration**:
```yaml
pretrained:
  video: "runs/video_classifier/best.pt"
  audio: "runs/audio_classifier/best.pt"
  load_heads_only: false  # Load everything
```

**Shell script**:
```bash
PRETRAINED_VIDEO="runs/video_classifier/best.pt"
PRETRAINED_AUDIO="runs/audio_classifier/best.pt"
LOAD_HEADS_ONLY="false"
```

**Advantages**:
- Starts from exactly your trained models
- Backbones already fine-tuned on your data

**Disadvantages**:
- Loses general-purpose capabilities of original YOLO/Whisper
- May overfit if your single-modal data was limited
- Generally not recommended unless you have a specific reason

---

## 📊 Comparison Table

| Strategy | YOLO Backbone | Whisper Encoder | Video Head | Audio Heads | Fusion | Recommended |
|----------|--------------|-----------------|------------|-------------|--------|-------------|
| **Default** | Pretrained (Ultralytics) | Pretrained (OpenAI) | Random | Random | Random | ⭐ First time |
| **Heads Only** | Pretrained (Ultralytics) | Pretrained (OpenAI) | From training | From training | Random | ⭐⭐⭐ Best |
| **Complete** | From training | From training | From training | From training | Random | ⚠️ Rare cases |

---

## 🔧 Implementation Details

### What gets loaded (Heads Only mode)

**Video model**:
- ❌ Skipped: `feature_extractor.*` (YOLO backbone)
- ✅ Loaded: `temporal_aggregation.*` (LSTM/attention)
- ✅ Loaded: `head.*` (classification head)

**Audio model**:
- ❌ Skipped: `encoder.*` (Whisper encoder)
- ✅ Loaded: `feature_projection.*` (projection layers)
- ✅ Loaded: `event_head.*` (event classification)
- ✅ Loaded: `vocalization_head.*` (if present)
- ✅ Loaded: `energy_head.*` (if present)

### Python API

```python
from synchronai.models.multimodal import MultiModalSynchronyModel

model = MultiModalSynchronyModel(
    video_config={...},
    audio_config={...},
    fusion_config={...}
)

# Strategy 1: Default (no action needed)
# Backbones are pretrained automatically

# Strategy 2: Load heads only (RECOMMENDED)
model.load_pretrained(
    video_ckpt="runs/video_classifier/best.pt",
    audio_ckpt="runs/audio_classifier/best.pt",
    load_heads_only=True  # Keep pretrained backbones
)

# Strategy 3: Load complete models
model.load_pretrained(
    video_ckpt="runs/video_classifier/best.pt",
    audio_ckpt="runs/audio_classifier/best.pt",
    load_heads_only=False  # Load everything
)
```

---

## 💡 Recommendations

### For best results:

1. **First**: Train separate video and audio classifiers
   ```bash
   bash scripts/bsub/pre_video_synchrony_bsub.sh
   bash scripts/bsub/pre_audio_synchrony_bsub.sh
   ```

2. **Then**: Use **Strategy 2 (Heads Only)** for multi-modal training
   ```yaml
   pretrained:
     video: "runs/video_classifier/best.pt"
     audio: "runs/audio_classifier/best.pt"
     load_heads_only: true
   ```

3. **Why this works**:
   - YOLO/Whisper backbones are extremely robust (trained on huge datasets)
   - Your heads are adapted to synchrony/event detection
   - Fusion learns to combine both modalities optimally
   - Two-stage training prevents overfitting

### Training time comparison:

| Strategy | Stage 1 (5 epochs) | Stage 2 (45 epochs) | Notes |
|----------|-------------------|---------------------|-------|
| Default | ~2-3 hours | ~18-27 hours | Heads learn from scratch |
| Heads Only | ~1-2 hours | ~9-18 hours | Faster convergence |
| Complete | ~1-2 hours | ~9-18 hours | Similar to heads only |

*Times are estimates on 40-core cluster with batch_size=16*

---

## ❓ FAQ

**Q: What if I only have a trained video model, not audio?**

A: Use heads only for video, default for audio:
```yaml
pretrained:
  video: "runs/video_classifier/best.pt"
  audio: null
  load_heads_only: true
```

**Q: Can I use different fusion types with pretrained models?**

A: Yes! The fusion module is always trained from scratch regardless of strategy.

**Q: Will this work with different YOLO/Whisper versions?**

A: Yes, as long as:
- Video config matches your trained model (e.g., yolo26s)
- Audio config matches your trained model (e.g., large-v3)

**Q: What if my checkpoints are in a different format?**

A: The code handles multiple formats:
- `checkpoint['model_state_dict']`
- `checkpoint['state_dict']`
- Direct state dict

**Q: Should I freeze backbones in stage 1?**

A: Yes! The two-stage training always:
- Stage 1: Freeze backbones (whether pretrained or loaded)
- Stage 2: Unfreeze with differential LRs

This pattern works for all strategies.

---

## 🚀 Quick Start

**Recommended workflow**:

```bash
# 1. Train video classifier
bash scripts/bsub/pre_video_synchrony_bsub.sh

# 2. Train audio classifier
bash scripts/bsub/pre_audio_synchrony_bsub.sh

# 3. Edit multi-modal config
vim scripts/train_multimodal_synchrony.sh
# Set:
#   PRETRAINED_VIDEO="runs/video_classifier/best.pt"
#   PRETRAINED_AUDIO="runs/audio_classifier/best.pt"
#   LOAD_HEADS_ONLY="true"

# 4. Train multi-modal
bash scripts/bsub/pre_multimodal_synchrony_bsub.sh
```

That's it! You'll get the best of pretrained backbones + task-adapted heads + learned fusion.

# Multi-Modal Heatmap Visualization

The multi-modal training supports automatic heatmap generation during training, showing how video, audio, and fusion predictions evolve over time.

## What Gets Visualized

Each heatmap shows **4 timelines** for a sample video clip:

1. **Video-Only Predictions**: What the YOLO pathway predicts alone
2. **Audio-Only Predictions**: What the Whisper pathway predicts alone
3. **Fused Predictions**: What the combined multi-modal model predicts
4. **Probability Timeline**: Line plot comparing all three modalities

### Example Heatmap Output

```
runs/multimodal_classifier/heatmaps/epoch_0010/
├── sample_video_comparison.png  # Visual comparison
└── sample_video_data.json        # Raw predictions
```

## How to Enable

### Option 1: Config File

Edit [configs/train/multimodal_classifier.yaml](../../configs/train/multimodal_classifier.yaml):

```yaml
training:
  heatmap_epoch_interval: 5  # Generate every 5 epochs
  heatmap_video_path: "path/to/sample/video.mp4"  # Optional (auto-selected if not provided)
```

### Option 2: Shell Script

Edit [scripts/train_multimodal_synchrony.sh](../../scripts/train_multimodal_synchrony.sh):

```bash
HEATMAP_EPOCH_INTERVAL=5  # Generate every 5 epochs
HEATMAP_VIDEO="/path/to/sample/video.mp4"  # Optional
```

### Option 3: Command Line

```bash
python -m synchronai.training.multimodal.train \
    --labels-file data/labels.csv \
    --save-dir runs/multimodal_classifier \
    --heatmap-epoch-interval 5 \
    --heatmap-video /path/to/sample.mp4 \
    ...
```

## What You'll See

### Timeline Visualization

The heatmap shows temporal predictions across a 10-second clip:

```
┌─────────────────────────────────────────┐
│ Video-Only:  🟢🟢🟢🔴🔴🔴🟢🟢🟢🟢        │
│ Audio-Only:  🟢🔴🟢🔴🟢🔴🟢🟢🔴🟢        │
│ Fused:       🟢🟢🟢🟢🔴🔴🟢🟢🟢🟢        │
└─────────────────────────────────────────┘
     0s    2s    4s    6s    8s   10s

🟢 = Synchrony detected
🔴 = No synchrony
```

### Probability Timeline

Line plot showing:
- **Blue line**: Video-only probability
- **Orange line**: Audio-only probability
- **Green line**: Fused probability
- **Red dashed**: Threshold (0.5)

This helps you see:
- How fusion combines modalities
- Which modality is more confident
- Whether fusion is working as expected

## Insights from Heatmaps

### Good Fusion Behavior

```
Video:  ⬛⬛⬛🟩🟩🟩⬛⬛⬛⬛  (Detects visual cues)
Audio:  🟩⬛🟩⬛🟩⬛🟩🟩⬛🟩  (Detects vocalizations)
Fused:  🟩🟩🟩🟩🟩🟩🟩🟩⬛🟩  (Best of both!)
```

Fusion correctly combines evidence from both modalities.

### Poor Fusion (Needs More Training)

```
Video:  🟩🟩🟩⬛⬛⬛🟩🟩🟩🟩
Audio:  🟩⬛🟩⬛🟩⬛🟩🟩⬛🟩
Fused:  ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛  (Worse than individual!)
```

Fusion is not working - may need:
- More training epochs
- Different fusion type (try `gated` instead of `cross_attention`)
- Better pretrained heads

### Video-Dominant

```
Video:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
Audio:  ⬛🟩⬛🟩⬛🟩⬛⬛🟩⬛
Fused:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩  (Matches video)
```

Model learned to rely mostly on video pathway.

### Audio-Dominant

```
Video:  ⬛🟩⬛🟩⬛🟩⬛⬛🟩⬛
Audio:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩
Fused:  🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩  (Matches audio)
```

Model learned to rely mostly on audio pathway.

## Performance Considerations

- Heatmap generation pauses training briefly (~30 seconds for 10s clip)
- Only generate every 5-10 epochs to avoid slowdown
- Heatmaps are saved to disk (a few MB per epoch)
- Use for debugging/monitoring, disable for production training

## Recommended Settings

### Debugging / Development

```yaml
heatmap_epoch_interval: 5  # Frequent updates
```

Good for:
- Understanding model behavior
- Catching issues early
- Visualizing learning progress

### Production Training

```yaml
heatmap_epoch_interval: 10  # Less frequent
# or
heatmap_epoch_interval: 0   # Disabled
```

Good for:
- Faster training
- Saving disk space
- Final training runs

## Auto-Selection of Video

If you don't specify `heatmap_video_path`, the system will:

1. Extract first video path from training dataset
2. Use it for all heatmap generations
3. Log the selected video:
   ```
   Auto-selected heatmap video: /path/to/sample.mp4
   ```

## File Formats

### PNG Image

Visual comparison plot showing all modalities and timeline.

### JSON Data

```json
{
  "predictions": [
    {
      "second": 0,
      "video_probability": 0.8234,
      "audio_probability": 0.6123,
      "fused_probability": 0.8967
    },
    ...
  ],
  "statistics": {
    "video_mean_prob": 0.7234,
    "audio_mean_prob": 0.5891,
    "fused_mean_prob": 0.7891
  }
}
```

Can be loaded for further analysis:

```python
import json

with open('heatmaps/epoch_0010/sample_data.json') as f:
    data = json.load(f)

# Analyze predictions
for pred in data['predictions']:
    print(f"Second {pred['second']}: "
          f"Fused={pred['fused_probability']:.2f} "
          f"(Video={pred['video_probability']:.2f}, "
          f"Audio={pred['audio_probability']:.2f})")
```

## Troubleshooting

### "Heatmap video not found"

**Solution**: Check video path exists and is accessible:
```bash
ls -lh /path/to/video.mp4
```

### "Heatmap generation failed"

**Common causes**:
- Video file corrupted
- Audio extraction failed (need `ffmpeg` installed)
- Out of disk space

**Solution**: Check logs for specific error, ensure `ffmpeg` is available:
```bash
ffmpeg -version
```

### Heatmaps look wrong

**Possible issues**:
1. **All predictions are 0.5**: Model not trained yet (early epochs)
2. **All same color**: Threshold too high/low, adjust in config
3. **Fusion worse than individual**: Needs more training or different fusion type

## Example Workflow

```bash
# 1. Enable heatmaps in config
vim configs/train/multimodal_classifier.yaml
# Set: heatmap_epoch_interval: 5

# 2. Train model
bash scripts/bsub/pre_multimodal_synchrony_bsub.sh

# 3. Monitor heatmaps during training
ls -lh runs/multimodal_classifier/heatmaps/

# 4. View latest heatmap
open runs/multimodal_classifier/heatmaps/epoch_0010/sample_video_comparison.png
```

## Comparison with Video-Only Heatmaps

| Feature | Video-Only Heatmaps | Multi-Modal Heatmaps |
|---------|---------------------|----------------------|
| **Modalities** | Video pathway only | Video + Audio + Fusion |
| **Timeline** | Single timeline | 3 timelines + comparison |
| **Grad-CAM** | Optional spatial heatmaps | Not yet implemented* |
| **Insights** | Visual attention | Modality fusion behavior |

*Grad-CAM for multi-modal models is more complex and not currently supported, but could be added in the future.

## Tips for Interpretation

1. **Early training (epochs 1-5)**:
   - Expect random/unstable predictions
   - Fusion may be worse than individual modalities
   - This is normal - give it time!

2. **Mid training (epochs 10-25)**:
   - Fusion should start outperforming individual modalities
   - Predictions become more stable
   - Look for complementary patterns (video catches visual, audio catches vocalization)

3. **Late training (epochs 30-50)**:
   - Fusion should consistently outperform
   - High confidence on clear examples
   - Appropriate uncertainty on ambiguous cases

4. **Overfitting signs**:
   - Perfect predictions on training video
   - Very high confidence (all >0.95 or <0.05)
   - May need more regularization or data augmentation

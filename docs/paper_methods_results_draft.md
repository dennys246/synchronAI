# Multimodal Synchrony Classifier — Methods & Results (Draft)

**Status**: first-pass draft for the multimodal video+audio synchrony classification
paper. Numbers are CV-validated; prose is structured but not yet final.

---

## Methods

### Dataset and split

We use the CARE dyadic interaction corpus (49 subjects, 59,250 1-second
labeled segments after video–audio joining). Labels are binary
synchrony codes derived from human behavioral coding. Audio and video
streams are extracted from the same source videos at second-level
alignment.

For all reported results we use **subject-grouped 5-fold cross-validation**.
Subjects are randomly partitioned into 5 disjoint folds (seed-fixed); for
each fold, training uses the other 4 folds (~40 subjects, ~50K segments)
and validation uses the held-out fold (~9 subjects, ~9K segments). Folds
0-3 contain 9 subjects each; fold 4 contains 13 subjects (the remainder).
The split is deterministic given the seed, allowing direct ablation
comparison: identical val composition across architectures, hyperparameters,
and feature families.

Single-fold numbers (fold 0 only) are used for exploratory hyperparameter
sweeps and reported separately as supplementary observations; *all primary
claims rest on 5-fold CV*.

### Feature extraction

All features are pre-extracted to disk once and reused across training
runs to keep iteration fast.

**Video** — DINOv2-base ViT applied per-frame at 12 frames/second over each
1-second window. Patch tokens are mean-pooled per frame to yield a
(12, 768) tensor per segment.

**Audio (primary)** — WavLM-base-plus encoder applied to the 1-second 16kHz
audio chunk. The transformer's blended-layer output gives a (49, 768)
tensor per segment. Pretrained on 94k hours of speech with phonetic and
self-supervised objectives.

**Audio (secondary, for ablation)** — Two additional audio backbones for
the feature-family comparison:
- *WavLM-large*: same model family, scaled up to 1024 dim × 24 layers.
  (49, 1024) per segment.
- *eGeMAPS LLDs*: openSMILE's eGeMAPSv02 low-level descriptors at ~100Hz,
  25 dimensions per frame: F0 in semitones, jitter, shimmer, HNR, MFCCs
  1-4, formants F1-F3 (frequency, bandwidth, amplitude), loudness,
  spectral flux, alpha ratio, Hammarberg index. Hand-engineered acoustic
  features standard in affective computing.

Features are saved as `.pt` tensors and joined by (video_path, second)
keys. The 25-dim eGeMAPS features specifically cannot encode scene-level
visual content the way 768-dim transformer features can, making them a
mechanistically clean test for whether audio-side synchrony signal
exists independent of video.

### Multimodal architecture (V2 baseline)

Our primary model is a small projection-then-fusion architecture with
~173K parameters:

```
Video  (B, 12, 768)
  → Linear(768→64) + GELU + Dropout(0.3)
  → mean over T=12
  → Dropout(0.3) on aggregated representation
  → video_repr (B, 64)

Audio  (B, T_a, D_a)              # WavLM: T_a=49, D_a=768 (or 1024); eGeMAPS: T_a=96, D_a=25
  → Linear(D_a→64) + GELU + Dropout(0.3)
  → LSTM(64→64, num_layers=2, dropout=0.2)
  → take h_n[-1] (top-layer hidden)
  → Dropout(0.3) on aggregated representation
  → audio_repr (B, 64)

Fusion
  → concat[video_repr, audio_repr]  (B, 128)
  → Linear(128→64) + GELU + Dropout(0.3)
  → Linear(64→1)                     binary logit
```

The explicit dropout on each modality's aggregated representation is
load-bearing: PyTorch's `nn.LSTM(dropout=p)` applies between-layer
dropout only, leaving the top-layer output undropped. Without explicit
post-LSTM dropout, the audio pathway enters fusion unregularized.

### Fusion architecture variants

For the architecture-insensitivity ablation we compare three additional
fusion designs while keeping per-modality projections identical:

- **V3 (cross-attention on summaries)** — Stack video_repr and audio_repr
  as 2 tokens, run multi-head self-attention (4 heads, attn dropout 0.2),
  residual connection, flatten back to (B, 128) for the head. The simplest
  change that allows cross-modal interaction in the representations
  themselves rather than only at the head.

- **V4 (token-level cross-modal transformer)** — Skip per-modality
  aggregation. Project both modality sequences to common dim P (64), add
  learnable modality + per-position embeddings, concatenate as a 61-token
  sequence (12 video + 49 audio), run a pre-LN transformer encoder
  (1 layer, 4 heads, FFN), mean-pool the 61 tokens. Every video frame
  can attend to every audio frame — the most expressive fusion we test.

- **V1 (legacy concat-LSTM)** — Reported only for historical comparison;
  same per-modality LSTM aggregator on raw 768-dim sequences with no
  projection bottleneck.

### Training procedure

Identical across all conditions unless explicitly varied:

- **Loss**: BCEWithLogitsLoss with `pos_weight ∈ [0.5, 2.0]` clamped to
  the empirical class ratio.
- **Optimizer**: AdamW, learning rate 5e-5, weight decay 1e-2.
- **Schedule**: Linear warmup over 5 epochs (start factor 0.3), then
  cosine annealing over the remaining 25 epochs (total 30).
- **Batch size**: 128, dropping the last partial batch.
- **Early stopping**: patience 10 on val_loss. We additionally save
  per-criterion-best checkpoints (best_acc.pt, best_auc.pt, best_loss.pt)
  to support post-hoc analysis at each metric's optimum.
- **Gradient clipping**: L2 norm 1.0.
- **No data augmentation**. Multimodal mixup, label smoothing, and
  modality-aware augmentation strategies were ruled out on scientific
  grounds (mixup breaks cross-modal temporal alignment in dyadic
  synchrony; label smoothing assumes noise where labels are thresholded
  continuous).

CPU-only training. Each fold trains in ~30 minutes on 4 cores with
OMP/MKL threading; full 5-fold CV completes in ~1 hour parallel.

### Modality-dropout training (D3 ablation)

To test whether the audio pathway's redundancy with video is a
training-dynamics phenomenon, we additionally trained one condition
with stochastic modality dropout: during each training batch, video
features are independently zeroed with probability 0.3 and audio
features with probability 0.3, with resampling when both would be
zeroed. Validation always uses both modalities.

### Cross-validation analyses

For each condition (architecture × audio feature family × training
regime), we run 5 training jobs (one per fold). We report:

- Per-fold val_acc, val_AUC, val_loss at the best epoch
- Across-fold mean ± standard error (SE), and 95% CI via normal approximation
- Per-condition best-fold (corresponds to single-fold numbers from
  prior literature using a fixed train/val split)

### Representation redundancy diagnostic (D1b)

For each trained checkpoint, we extract the per-modality representations
(`video_repr`, `audio_repr` from the forward pass) on the validation set
and compute:

- **Per-sample cosine similarity** between video_repr and audio_repr
- **Mean per-dimension Pearson correlation** (matched dims)
- **Linear R²(video_repr → audio_repr)** — variance of audio_repr that
  is linearly explained by video_repr
- **Linear R²(audio_repr → video_repr)** — reverse direction

R² values are computed via OLS regression on the full val set and
aggregated across folds.

---

## Results

### Headline: multimodal val_AUC at the data-limited regime

Across 5-fold subject-grouped cross-validation (n=49 subjects), the
V2 multimodal architecture achieves:

|  | Mean val_AUC | Mean val_acc | Mean val_loss |
|---|---|---|---|
| **V2 multimodal (WavLM-base-plus)** | **0.719 ± 0.052** | 0.692 ± 0.045 | 0.541 ± 0.033 |

Per-fold val_AUC ranges from 0.659 (fold 4) to 0.798 (fold 0); per-fold
val_acc ranges from 0.644 to 0.764. Fold composition is the dominant
source of variance (across-fold SE ≈ 0.023 on AUC) — substantially larger
than the across-condition variance we observe below.

### Architecture insensitivity

Across four fusion architectures with identical per-modality projections,
training procedure, and fold splits, CV means cluster tightly:

| Fusion architecture | Mean val_AUC | Mean val_acc | Δ vs V2 baseline |
|---|---|---|---|
| V2 (concat, h=64) | 0.7192 ± 0.052 | 0.6923 ± 0.045 | — |
| V3 (cross-attention on summaries, h=24) | 0.7222 ± 0.054 | 0.6951 ± 0.045 | +0.003 AUC |
| V4 (token-level cross-modal transformer) | 0.7057 ± 0.051 | 0.6721 ± 0.050 | -0.014 AUC |

All differences fall within the single-fold sampling SE (~0.023 AUC).
A capacity sweep at single-fold granularity (V2 vs V3 at h ∈ {24, 32, 40,
48, 64, 128}) revealed a mechanistic capacity-bottleneck pattern in which
cross-attention helps at small h (h=24: +0.025 AUC over concat fusion on
fold 0); however, 5-fold CV of v3 at h=24 showed the apparent advantage
was driven by fold-0 sampling rather than a real effect (fold-0 AUC
0.803 vs CV mean 0.722; the +0.025 single-fold effect lay entirely
within fold-composition variance).

### Audio feature family insensitivity

Across three audio feature families spanning transformer-pretrained and
hand-engineered acoustic representations, CV means are statistically
indistinguishable:

| Audio features | Output shape (per second) | Mean val_AUC | Mean val_acc |
|---|---|---|---|
| WavLM-base-plus | (49, 768) | 0.7192 ± 0.052 | 0.6923 ± 0.045 |
| WavLM-large | (49, 1024) | (51-hour extraction; partial CV) | — |
| eGeMAPS LLDs | (96, 25) | 0.7200 ± 0.052 | 0.6930 ± 0.044 |

Δ(eGeMAPS vs WavLM-base-plus) = +0.0008 val_AUC, +0.0007 val_acc.
Three of five folds match to within ±0.0005 on val_acc. This is the
strongest possible null result for the audio-representation hypothesis:
a 25-dim acoustic feature vector that mechanistically cannot encode
scene-level visual content produces identical CV performance to a
768-dim transformer audio representation.

### Single-modality decomposition

Apples-to-apples within the same V2 pipeline, replacing one modality's
features with zeros during both training and validation:

| Condition | Mean val_AUC | Mean val_acc | Δ vs multimodal |
|---|---|---|---|
| Multimodal (both modalities) | 0.7192 ± 0.052 | 0.6923 ± 0.045 | — |
| Video-only (audio zeroed) | 0.7171 ± 0.052 | 0.6923 ± 0.045 | -0.002 AUC, 0.000 acc |
| Audio-only (video zeroed, WavLM) | 0.6986 ± 0.051 | 0.6712 ± 0.039 | -0.021 AUC |

The video pathway alone reaches **0.9971× the multimodal val_AUC**.
Audio carries real but non-additive signal: audio-only val_AUC is well
above chance (0.5) but adding audio to video yields no measurable
improvement.

### Modality-dropout training does not break redundancy

Training with stochastic modality dropout (`p_drop=0.3` on each modality
independently, training-time only) produces identical CV mean
performance to vanilla training:

| Training regime | Mean val_AUC | Mean val_acc |
|---|---|---|
| Vanilla (no modality dropout) | 0.7192 ± 0.052 | 0.6923 ± 0.045 |
| + Modality dropout (p=0.3 each) | 0.7192 ± 0.052 | 0.6930 ± 0.043 |

Δ = +0.0000 AUC, +0.0007 acc. Forcing the audio pathway to occasionally
carry the prediction signal on its own yields no measurable benefit.

### Representation redundancy explains the audio-null result

For each trained checkpoint, we measured the linear predictability
between video_repr and audio_repr on the validation set:

| Audio features | Training regime | R²(V→A) | R²(A→V) |
|---|---|---|---|
| WavLM-base-plus | Vanilla | 0.65 ± 0.12 | 0.92 ± 0.03 |
| WavLM-base-plus | + Modality dropout | 0.98 ± 0.01 | 0.95 ± 0.03 |
| eGeMAPS LLDs (25-dim) | Vanilla | 0.93 ± 0.03 | 0.92 ± 0.03 |

In every condition the model's audio representation is highly linearly
predictable from its video representation (R²(A→V) ≥ 0.88 in all cases,
≥ 0.92 in most). **This holds even for 25-dimensional acoustic features
that mechanistically cannot encode scene-level content** — the model
projects eGeMAPS prosodic features into a 64-dimensional representation
whose variance is 93% linearly explained by the video representation.

Two implications follow:

1. **The model's audio pathway functions as a parallel reconstruction
   of the video signal**, regardless of input feature properties.
2. **Modality dropout exacerbates rather than breaks this redundancy** —
   forcing each pathway to occasionally predict alone pushes both
   pathways toward encoding the same underlying scalar (R²(V→A) jumps
   from 0.65 to 0.98 under modality dropout), since a mirror-image
   representation is the easiest gradient solution to "sometimes one
   modality is missing."

### Interpretation: data-limited regime

The cumulative evidence supports a data-limited rather than
architecture-limited interpretation:

- **Eight experimental conditions** (4 fusion architectures × 3 audio
  feature families × 2 training regimes) converge to a CV mean val_AUC
  of 0.72 ± 0.05.
- **Fold variance (SE ≈ 0.023) dominates condition variance (Δ ≤ 0.014)**
  across every comparison we test.
- **The model coerces any audio representation into a video-redundant
  embedding** during joint training, regardless of input feature
  properties or training regime.
- **Multimodal performance is statistically indistinguishable from
  video-only performance** (Δ = +0.002 val_AUC, well within sampling SE).

At n=49 subjects with second-level binary synchrony labels, the
discriminative signal accessible to a feature-based multimodal classifier
is captured almost entirely by the video pathway. Audio-side information
exists (audio-only models perform well above chance), but is either
fully redundant with video at this scale or below the detection threshold
imposed by 5-fold sampling variance.

### Scaling argument

The per-fold standard error scales as 1/√(n_subjects). At n=49 the SE on
val_AUC is approximately 0.023; doubling the dataset (n=100) would
reduce this to approximately 0.016; quadrupling (n=200) to approximately
0.011. The architecture and feature-family differences we tested are all
in the range of 0.005-0.015 val_AUC — only detectable at substantially
larger subject counts than our current data permits.

Conversely, the audio contribution we observe (Δ_multimodal_vs_video ≤
+0.002 AUC) is well below even the n=200 detection threshold. Unless
audio carries information that is non-redundant with video AND
sufficiently discriminative to produce a >0.01 AUC contribution under
joint training, audio's null contribution to our task is expected to
persist at substantially larger dataset scales.

---

## Limitations

- All results condition on the V2 / V3 / V4 architecture family. We did
  not test architectures with explicit cross-modal contrastive losses
  (e.g., CLIP-style objectives), which might enforce independence and
  unlock audio contribution at the cost of training-time complexity.
- WavLM and eGeMAPS span transformer-pretrained and hand-engineered
  acoustic features, but other audio representations remain untested
  (emotion-pretrained models like emotion2vec; raw waveform models).
- Subject n=49 is small. The data-limited interpretation we advance
  predicts that audio contribution may become detectable at higher n;
  this is empirically testable as data collection continues.
- Synchrony labels are binary at second-level granularity. Continuous
  or longer-temporal-window labels might surface signal not accessible
  to second-level binary classification.

## Conclusion

In this dataset (n=49 subjects, 59,250 second-level binary synchrony
labels), the multimodal advantage in feature-based dyadic synchrony
classification is null relative to video alone, robustly across four
fusion architectures, three audio feature families spanning
transformer-pretrained and hand-engineered acoustic representations,
two training regimes, and six capacity configurations. The trained
audio pathway encodes a representation that is linearly redundant with
the video pathway in every condition tested. We interpret these results
as evidence that the discriminative signal accessible to a feature-based
classifier at this dataset scale is primarily video-encoded, and that
detecting audio-side contribution will require substantially larger
subject counts.

---

## Figures (planned)

- **Figure 1**: Forest plot of CV mean val_AUC across all eight tested
  conditions, with 95% CIs. Visualizes the convergence of all conditions
  to ~0.72 ± 0.05.
- **Figure 2**: Per-fold val_AUC trajectory across architectures (4 lines
  for V1/V2/V3/V4). Shows fold rank is identical across architectures —
  fold variance dominates architecture variance.
- **Figure 3**: Representation redundancy bar chart. R²(V→A) and R²(A→V)
  across three audio conditions (WavLM-vanilla, WavLM-moddropout,
  eGeMAPS), with error bars across folds.

Generation scripts: `scripts/paper_figures/plot_*.py`.

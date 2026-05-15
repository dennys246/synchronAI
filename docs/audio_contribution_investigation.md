# Audio Contribution Investigation

**Status**: planning. Triggered by the 5-fold CV ablation result showing
WavLM-base-plus audio contributes ~zero to multimodal performance.

## The trigger finding

| Config | CV mean acc | CV mean AUC | Δ vs video-only |
|---|---|---|---|
| Audio-only (video zeroed) | 0.6712 ± 0.039 | 0.6986 ± 0.051 | -0.019 AUC |
| Video-only (audio zeroed) | 0.6923 ± 0.045 | 0.7171 ± 0.052 | baseline |
| Multimodal (both) | 0.6923 ± 0.045 | 0.7192 ± 0.052 | **+0.002 AUC** |

Per-fold val_acc was **identical to 4 decimals** between video-only and multimodal
(0.6923 = 0.6923). Adding WavLM-base-plus audio to DINOv2 video changes effectively
zero predictions across the val set.

## Why this is a red flag, not a finding

Audio alone reaches 0.699 AUC — **well above chance** (0.50). So WavLM-base-plus features
carry real signal for this task. When the model is forced to rely on audio (video zeroed),
it extracts ~0.20 AUC above chance.

But when both modalities are available, the audio pathway contributes nothing.
That means the model is either (a) not using audio at all, (b) using it but its
contribution is fully redundant with video, or (c) being prevented from using audio
by some architectural or training-dynamics issue.

For a dyadic-synchrony task — where speech timing, prosody, and turn-taking are
plausibly informative — zero contribution is surprising enough to warrant
investigation before accepting it as a real finding.

## Four hypotheses

| # | Hypothesis | Test |
|---|---|---|
| H1 | The head's first Linear layer assigns near-zero weights to the audio half of its input → model functionally ignores audio | Diagnostic 1: head weight inspection |
| H2 | Blended WavLM-base-plus features (49, 768) collapse the prosodic/timing information that synchrony depends on; per-layer features preserve it | Diagnostic 2: train on per-layer WavLM features with learnable layer weights |
| H3 | Training dynamics: video gradients dominate early, head converges to "video-only" attractor before audio can contribute | Diagnostic 3: stochastic video-modality dropout during training (force audio pathway to remain useful) |
| H4 | WavLM features genuinely don't capture synchrony cues. Phonetic/semantic objectives in pretraining miss prosody-level features | Extract a different audio representation (emotion2vec, openSMILE prosodic features) |

## Diagnostic order — cheapest first

**D1 → D2 → D3 → H4 if all negative.** Each step rules out cheaper hypotheses
before committing to expensive feature extraction.

### Diagnostic 1: Head weight inspection (~10 min)
For each `v2_baseline_v6_cv5/fold_*/best.pt`:
- Load the head's first Linear layer weight, shape `(head_hidden, 2 * proj_dim)`.
- Split each row into video weights `[:, :proj_dim]` and audio weights `[:, proj_dim:]`.
- Compute Frobenius norms of each block. Compute ratio `||W_video|| / ||W_audio||`.
- Interpret:
  - **Ratio ≈ 1.0**: head treats modalities equally. Audio info is genuinely redundant with video → H2/H4 likely.
  - **Ratio > 5**: head ignores audio. Training dynamics or architecture problem → H1/H3 likely.

### Diagnostic 2: Per-layer WavLM features (~3-4 hours total)
Features already extracted at `data/wavlm_baseplus_perlayer_features/` ((13, 49, 768)).
Add learnable softmax layer weights to the audio pathway: `a = sum_k(w_k * a_k)` where
`w_k = softmax(learnable_logits)`. Then proceed through the usual projection + LSTM.
5-fold CV. If it clears the multimodal baseline (>0.72 AUC), blended was the issue.

### Diagnostic 3: Modality dropout (~2-3 hours)
Add per-batch stochastic video zeroing during training (e.g., 30% of batches see
audio only). Forces the model to maintain a useful audio pathway. 5-fold CV with
the same v2 architecture + dropout schedule. If this restores multimodal benefit,
H3 is the cause.

### H4 (escalation): Alternate audio representation
Extract emotion2vec or openSMILE prosodic features. Re-run CV. Only justified if D1-D3
all negative.

## What "success" looks like for the investigation

The cleanest outcome is that **one diagnostic clearly explains the null result** and
points to a fix. E.g.:
- D1 shows ratio = 10× → multimodal model ignores audio → D3 confirms training-dynamics
  fix → V5 architecture with modality dropout → re-run CV → audio contributes meaningfully.

The runner-up outcome is **all diagnostics negative** → WavLM-base-plus is genuinely
wrong for synchrony → extract alternate audio rep → re-test. Burns ~1-2 days but
informed by ruling out cheaper explanations.

## What NOT to do

- Don't accept "audio doesn't help" as the final result before completing D1-D3.
- Don't extract a new audio representation (H4) before D1-D3 finish. Risk: same null
  result with no diagnostic value.
- Don't tune fusion architecture further. The 4-architecture sweep already showed
  fusion-architecture-insensitivity at this dataset size.

# Multimodal Classifier v2 Plan

**Status**: planning. v1 produced a real-but-overfitting result; v2 targets convergence and a higher accuracy bar (~80%).

---

## Context: what v1 produced and why we're moving on

v1 architecture: DINOv2-base meanpatch features (12, 768) + WavLM-base-plus features (49, 768), each fed through `nn.LSTM(D→64)`, last hidden state taken, concat into 128-dim, MLP head to 1 logit.

v1 result (best epoch only): **val AUC 0.7616, val acc 0.702** at epoch 1 — beats best single-modality (video 0.697, audio 0.674) by ~6 AUC points. Saved at `runs/multimodal_features/lstm_concat/best.pt`.

v1 trajectory was textbook overfitting:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC | LR |
|---|---|---|---|---|---|---|
| 1 | 0.559 | 0.665 | 0.526 | 0.702 | **0.762** | 5.3e-5 (warmup) |
| 2 | 0.506 | 0.739 | 0.548 | 0.679 | 0.720 | 7.7e-5 |
| 3 | 0.482 | 0.754 | 0.572 | 0.672 | 0.675 | 1.0e-4 (peak) |
| 4 | 0.467 | 0.764 | 0.643 | 0.664 | 0.583 | 1.0e-4 |
| 5 | 0.457 | 0.770 | 0.563 | 0.695 | 0.701 | 9.96e-5 |

Train monotonically improved, val collapsed once warmup ended. **Peak val AUC came at the lowest LR of the entire run.** Strong signal that the model can learn but the optimization is destabilizing it.

v1 was also too slow (~11s/batch CPU, ~2.5h/epoch) due to recurrent matmuls on 768-dim sequences and LSF slot fragmentation across 4 hosts.

---

## Diagnoses (root causes, not symptoms)

### D1. Regularization gap on the recurrent layers
`nn.LSTM(num_layers=1)` ignores its `dropout` argument (PyTorch only applies inter-layer dropout when `num_layers > 1`). Combined with `dropout=0.3` only being applied in the head, the LSTMs see zero stochastic regularization on a 50K-sample training set. Once warmup ends, they memorize subject-level noise.

### D2. LR is too high for this scale and split
Best AUC was at warmup LR (5.3e-5). Peak LR (1e-4) destroyed it. The split is unforgiving — only 9 val subjects, so generalization is measured strictly across subjects. A higher LR overshoots the narrow basin where cross-subject features matter more than subject-specific ones.

### D3. No projection before recurrent aggregation
Audio sweep v2 winners all used `project_dim=32` to compress 768→32 before any temporal aggregation. We didn't. This costs:
- **~10× compute** (recurrent matmul on 768 instead of 32-64)
- **Less effective regularization** (a learnable bottleneck IS regularization)

### D4. Subject-level features dominate
50,127 train samples but only 40 train subjects = avg 1,253 samples per subject. The LSTMs have plenty of data per subject and start picking up subject-id-correlated features. Without explicit regularization, they prefer "memorize the 40 training subjects" to "learn what synchrony looks like".

### D5. Compute budget mismatch
2.5h/epoch × 50 epochs = 5+ days. Iteration loop is too slow to do meaningful HPO. We need ~10-20 min/epoch to make this practical.

---

## Goals for v2

| Goal | Target | Why |
|---|---|---|
| Val AUC | **≥ 0.80** | Publication bar |
| Val Acc | ≥ 0.78 | Reads cleaner than AUC for the paragraph |
| Generalization gap | Train-Val acc < 5% at convergence | Prevents v1-style memorization |
| Iteration speed | ~10-20 min/epoch | Enables HPO and ablations |
| Wall clock for full run | < 12 hours | One overnight |

---

## Proposed v2 architecture

### Default config (v2-baseline)

```
Video: (B, 12, 768)
  → Linear(768 → 64) + GELU + Dropout(0.3)        # projection bottleneck
  → mean over T=12                                  # video is short; LSTM not needed
  → Dropout(0.3)                                    # explicit reg on aggregated repr
  → (B, 64) video_repr

Audio: (B, 49, 768)
  → Linear(768 → 64) + GELU + Dropout(0.3)        # projection bottleneck
  → LSTM(64 → 64, num_layers=2, dropout=0.2)      # 2 layers so dropout actually fires
  → take h_n[-1, :, :]                              # top-layer hidden, NOT batch-first
  → Dropout(0.3)                                    # explicit reg on aggregated repr
  → (B, 64) audio_repr

Fusion:
  → concat(video_repr, audio_repr)                # (B, 128)
  → Linear(128 → 64) + GELU + Dropout(0.3)
  → Linear(64 → 1)                                 # binary logit
```

Estimated params: ~172K (down from 435K in v1). Estimated per-batch cost: ~1s instead of 11s.

**Why explicit `Dropout` after the aggregated reprs?** PyTorch's `nn.LSTM(dropout=p)` applies dropout *between* layers only — the top layer's output (`h_n[-1]`) is undropped. Without an explicit `Dropout` on `audio_repr`, it enters fusion unregularized. Same for video mean-pool. These are the single most important regularization sites for the small validation set.

### Default training config

| Knob | v1 | v2 |
|---|---|---|
| Peak LR | 1e-4 | **5e-5** |
| Warmup epochs | 3 | 5 |
| Weight decay | 1e-3 | **1e-2** |
| Dropout (LSTM inter-layer) | 0 (no-op) | **0.2** (real, on 2-layer LSTM) |
| Dropout (output reprs) | none | **0.3** (NEW — on aggregated video/audio reprs) |
| Dropout (head) | 0.3 | 0.3 |
| Batch size | 64 | 128 (fewer batches/epoch, larger gradient signal-to-noise) |
| Patience | 15 | 10 |
| Epochs | 50 | 30 |
| Mixup alpha | 0 (off) | 0 (deferred to sweep variant — see below) |

### Why mixup is NOT in baseline

The baseline already changes the architecture (projection bottleneck, 2-layer LSTM, explicit aggregated-repr dropout), the LR (5e-5), the weight decay (1e-2), and the batch size (128). Adding mixup on top means five overlapping regularizers added simultaneously after diagnosing one bug — we couldn't attribute the delta. Mixup interacts with `BCEWithLogitsLoss(pos_weight=...)` and with the soft-label trick non-trivially. Run baseline clean, then add mixup as a sweep variant (`v2_mixup`) to isolate its contribution.

---

## Sweep matrix for v2

Run after v2-baseline shows reasonable convergence (>0.75 val AUC, no overfit collapse).

Priority order:

| Order | Variant | What changes | Hypothesis |
|---|---|---|---|
| 1 | `v2_baseline_v6` | re-run with `--early-stop-metric val_loss` + multi-criterion checkpoints | Isolates "did we just pick the wrong stopping epoch?" — most likely outcome of the first batch |
| 2 | `v2_higher_capacity` | hidden=128 throughout | Settles "are we underfitting?" |
| 3 | `v2_lower_capacity` | hidden=32 throughout | Direct test of D4 (subject memorization) — at 0.74 train samples per param, the model has ~2.5× less room to memorize than baseline |
| 4 | `v2_more_reg` | dropout=0.5, weight_decay=3e-2 | Push past the epoch-5 overfit cliff |
| 5 | `v2_lstm_video` | Replace video mean-pool with LSTM(64→64) | Test whether 12 frames benefit from temporal modeling |
| 6 | `v2_per_layer_audio` | Use `wavlm_baseplus_perlayer_features` with learnable layer weighting (audio_sweep_v2 winning pattern) | More expressive WavLM representation; features already extracted |
| 7 | `v2_mixup` | feature-space mixup, α=0.2 | Isolate mixup contribution after a clean baseline exists |
| 8 | `v2_label_smoothing` | targets ← y·(1-ε) + 0.5·ε with ε=0.1 | Cheap calibration improvement |
| 9 | `v2_swa` | Stochastic Weight Averaging over last 5 epochs | Robust to bouncy val curves |
| 10 | `v2_attention_audio` | Replace audio LSTM with single-layer self-attention pool | Only if LSTM ceilings out |
| 11 | `v2_cross_attention` | Replace concat with cross-attention between modality projections | Last resort; biggest architectural delta |

**First batch (run in parallel)**: variants 1-4 above. Each is ~1 hour on CPU with `span[hosts=1]` + `OMP_NUM_THREADS=8`, so all four complete in ~the time the original v1 run took. Submit via `scripts/bsub/submit_v2_sweep.sh`.

**Most likely outcome of the first batch**: variant 1 (`v2_baseline_v6` with val_loss stopping) lands at val_acc ~0.76-0.78 on `best_acc.pt` — meeting or nearly meeting the publication bar by simply picking the right epoch. If so, the architecture changes were sufficient and the rest of the matrix becomes a robustness check rather than a search.

**k-fold CV**: defer until a variant beats baseline by ≥0.02 AUC, then 5-fold the winner only. Don't k-fold every variant — kills the iteration loop.

---

## Open questions — RESOLVED

1. **WavLM-large extraction**: **WAIT.** 4-8h CPU extraction with no guarantee of >0.01 AUC gain. Only extract if v2-baseline + variants 1-4 plateau below 0.78 AUC.

2. **Per-layer WavLM features**: include as sweep variant `v2_per_layer_audio` (#4 in priority). Cheap, features already extracted.

3. **k-fold CV**: defer. Single-fold for HPO, 5-fold only on the eventual winner (after a variant beats baseline by ≥0.02 AUC).

4. **`span[hosts=1]` enough?**: NOT BY ITSELF. Combine with `OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8` exports + `torch.set_num_threads()` in the script. `span[hosts=1]` puts the 8 slots on one host, but PyTorch's default thread pool doesn't honor LSF slot count — must be told explicitly via OMP env vars, or it repeats the v1 "only 3 cores active" symptom.

5. **Early-stopping criterion**: use simple `patience=10` on val_AUC. The script already saves `best.pt` at peak val_AUC. Train-val gap and AUC-drop are useful as *log-line monitoring signals*, not as stop conditions — they're too noisy with 9 val subjects.

---

## What to keep from v1

- **Pre-extracted features pipeline** (`scripts/train_multimodal_from_features.py`) — joining logic, subject-grouped split, preload-to-RAM, history+plot saving. Solid foundation.
- **`dinov2_features_meanpatch` + `wavlm_baseplus_features`** as the default feature pair — joining is verified clean (0 dropped on either side).
- **BSub script** structure ([scripts/bsub/multimodal_from_features_bsub.sh](../scripts/bsub/multimodal_from_features_bsub.sh)) — preflight, absolute python path, env-var configurable. Just bump version and update default args.
- **`.detach()` on loaded features** — required, do not remove.
- **`num_workers=0`** when reading from RAM — required, do not bump.

## What to throw out from v1

- **Architecture**: replace the `nn.LSTM(768→64, 1 layer)` per modality with the v2 projection-then-aggregator pattern. Add explicit `Dropout(0.3)` on each aggregated repr (otherwise LSTM top-layer output is undropped).
- **LR=1e-4**: too aggressive for this scale. Drop to 5e-5.
- **Patience=15**: too generous for the slow iteration loop. Drop to 10.
- **`-n 8`** without `span[hosts=1]`: led to fragmentation. Add the span constraint **and** export `OMP_NUM_THREADS=8` / `MKL_NUM_THREADS=8` (PyTorch CPU thread pool doesn't honor LSF slot count automatically).

## v1 reproducibility — preserved

v1 argparse defaults stay pinned exactly as today (LR=1e-4, WD=1e-3, dropout=0.3, epochs=50, batch=64, warmup=3, patience=15, num_workers=4). Adding `--arch {v1,v2}` selects the model class. v2 hyperparam values get passed via CLI from the BSub script (which already does this via env vars). This means existing v1 invocations are bit-identical; only the new env-var defaults in the BSub script change behavior.

---

## Expected outcome

If D1+D2+D3 are the right diagnoses, v2-baseline should:
- Converge in 10-20 epochs (not 1)
- Hit val AUC 0.80-0.82
- Train-val acc gap stay below 5%
- Run in < 8 hours total (preload + train)

If v2-baseline plateaus below 0.78, the bottleneck is the data/representation, not the optimization, and we'd need WavLM-large extraction or attention-based fusion to push higher.

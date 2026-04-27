# Kickoff prompt for the multimodal v2 session

Paste the block below to start the next session. It assumes a fresh Claude with no conversation memory, and points it at the artifacts from this session so it can ramp up quickly.

---

I want to start a planning + refinement session for v2 of the multi-modal video+audio synchrony classifier. Don't write any code yet — I want to align on architecture and methodology first.

**Background**: We just finished a v1 run that produced a real result (val AUC 0.7616 at epoch 1) but overfit aggressively after that. The full diagnosis, the proposed v2 architecture, and the open questions are written up at `docs/multimodal_v2_plan.md`. Read that first — it's the brief for this session. Also read `docs/multimodal_v2_kickoff_prompt.md` (this file) so you have full context.

**The goal for v2**: get to ~80% accuracy / 0.80+ val AUC on the held-out subject split, in a training run that finishes in under 12 hours on CPU (no GPU available). Publication bar is ~80% accuracy.

**Constraints**:
- CPU-only (no GPU access).
- Pre-extracted features only — re-extracting raw modality features needs GPU.
- Available features are documented in the project memory under "Pre-extracted multimodal feature dirs and shapes" (see `MEMORY.md`). DINOv2 base-meanpatch and WavLM-base-plus are extracted; WavLM-large is not.
- Subject-grouped 80/20 split is non-negotiable (per `CLAUDE.md`).
- Follow the rest of `CLAUDE.md`'s agent principles: simplicity over ceremony, push back when warranted, no band-aids, scope discipline.

**What I want from this session, in order**:

1. **Verify the diagnosis.** Read `docs/multimodal_v2_plan.md` and the v1 history at `runs/multimodal_features/lstm_concat/history.json`. Tell me whether you agree D1-D5 are the right root causes. Push back if you see something I missed or disagree with one of them.

2. **Refine the proposed v2 architecture.** The plan proposes a projection-bottleneck + 2-layer LSTM (with real dropout) + concat fusion. Is this the right starting point? In particular: should video go through an LSTM or is mean-pool over 12 frames sufficient? Is `hidden=64` enough capacity? Are we under-regularizing or over-regularizing for 50K samples / 40 train subjects?

3. **Refine the sweep matrix.** Plan lists 7 variants. Which 2-3 should we run first after the baseline, and which can wait? Are there variants I missed (e.g., feature-level data augmentation, label smoothing, focal loss, ensemble)?

4. **Resolve the open questions.** Plan section "Open questions to resolve". Specifically: (a) is k-fold CV worth the 5× compute? (b) what's the right early-stopping criterion (currently I propose stopping if train-val acc gap >5% OR val AUC drops >0.02 from peak)? (c) should we extract WavLM-large now or wait?

5. **Produce a concrete implementation plan** for v2-baseline only (not the full sweep). I want a step-by-step list with file paths and the specific changes — no code yet, just the design. We'll implement next session after I review the plan.

**What NOT to do this session**:
- Don't write code. Plan only.
- Don't kick off any training runs.
- Don't restructure the v1 pipeline beyond what's needed for v2 — v1's data loading and BSub scaffolding work and should be reused.

Once you've read the plan and we've aligned on D1-D5 and the architecture, I want a written response covering points 1-5 above. Be terse but substantive. If something in my plan is wrong, say so.

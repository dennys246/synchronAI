## synchronAI agent notes

### Principles
- Keep changes minimal; reuse existing modules/folders when possible.
- Build so functionality and architecture can be used in unison or independently between multiple modalities (video/audio/fnirs).
- Creating larger systems is permissible to simplify handling of modalities however keep processes simple.
- Separate concerns: `data/` (I/O + preprocessing), `models/` (architectures), `training/` (loops), `inference/` (generation).
- fNIRS preprocessing must go through `hrfunc.preprocess_fnirs` (see `src/synchronai/data/fnirs/processing.py`).
- fNIRS models use **per-pair architecture** (feature_dim=2, one HbO/HbR pair per model). The legacy 20-channel format is archived.
- Prefer readable code over clever code: add short comments where intent isn't obvious.
- Import dependencies at module top; if a deferred import is required (runtime config or optional dependency), isolate it in a helper with a short reason.
- Log progress to the terminal for long-running steps (training loops + diffusion sampling).
- Use Hugging Face repos as the canonical model distribution source; never commit access tokens or private artifacts.

### BSub Script Versioning
- **Every BSub script change MUST increment `SCRIPT_VERSION`** — see CLAUDE.md for full rules.
- Every job must echo its version as the first output line so logs are traceable.
- This is non-negotiable: cluster debugging is impossible without knowing which script version ran.

### Cluster Conventions
- **PYTHONPATH, not pip install**: Set `PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"` instead of `pip install -e .` to avoid race conditions in shared NFS venvs.
- **Absolute paths**: All feature dirs, save dirs, and weight paths must use `$SYNCHRONAI_DIR/` prefixes. Relative paths fail inside Docker containers.
- **SCRIPT_VERSION**: Every BSub script defines and echoes a version string as the first output line.
- **Replace hardcoded usernames** with `$USER` (e.g., `/home/$USER/.bashrc`).

### Hugging Face Upload Principles
- Repos: `dennys246/fNIRS_diffusion`, `dennys246/fNIRS_synchrony`, `dennys246/video_synchrony`, `dennys246/audio_synchrony`, `dennys246/transcript_synchrony`
- Per-pair fNIRS models upload one model per pair (e.g., `S1-D1_HbO_HbR/`).
- Always upload model artifacts from a clean `--save-dir` (config + weights + any metadata).
- Access tokens via `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (or `--token` in CLI).
- Avoid uploading large caches or raw data; use allow/ignore patterns to keep the repo lean.

### Hugging Face Process (upload + download)
- Upload:
  - `python -m synchronai.utils.hf_hub upload --model fnirs_diffusion --path runs/fnirs_diffusion`
- Download:
  - `python -m synchronai.utils.hf_hub download --model fnirs_diffusion --path runs/fnirs_diffusion`

### Quality Tier System (fNIRS)
- **Gold**: Clean signal, minimal motion artifacts — used for training and holdout.
- **Standard**: Acceptable signal with minor artifacts — used for training.
- **Salvageable**: High motion but recoverable — used only for holdout monitoring (not training).
- **Rejected**: Unusable signal — excluded entirely.
- Tiers are computed during feature extraction and stored in the `quality_tier` column of `feature_index.csv`.
- Training: `--include-tiers "gold,standard"`. Holdout: `--holdout-tiers "gold,salvageable"`.

### Video Workflow (DINOv2)
DINOv2-small is the primary video backbone (YOLO is legacy).

1. **Feature extraction**: Extract DINOv2 frame embeddings from video clips → save to disk.
2. **Sweep training**: Train temporal aggregation (LSTM) + MLP head over pre-extracted features.
3. **Two-stage fine-tuning**: Stage 1 = frozen DINOv2, train head. Stage 2 = unfreeze with differential LRs (head LR = backbone_lr × 5).

### Audio Workflow (WavLM)
WavLM-base-plus is the primary audio encoder (Whisper is legacy).

1. **Feature extraction**: Extract WavLM per-layer embeddings from audio segments → save to disk.
2. **Sweep training**: Train classification head over pre-extracted features.
3. **Two-stage fine-tuning**: Stage 1 = frozen WavLM, train head. Stage 2 = unfreeze with differential LRs.

### fNIRS Workflow (Per-Pair Generative Pretraining)
Per-pair DDPM diffusion model (TensorFlow, feature_dim=2).

1. **Generative pretraining**: Train one diffusion model per fNIRS pair on preprocessed hemoglobin data.
   - `python scripts/generative_pretrain.sh` (or BSub variant)
2. **Feature extraction**: Generate synthetic windows + extract latent features from trained diffusion models → `feature_index.csv`.
   - `python scripts/extract_fnirs_features.py --data-dir <PATH> --save-dir <FEATURE_DIR>`
3. **Classifier training**: Train synchrony classifier on extracted features.
   - `python scripts/train_fnirs_from_features.py --feature-dir <FEATURE_DIR>`
4. **Artifacts** under `--save-dir`:
   - `fnirs_diffusion_config.json`: model/data metadata (pair names, sfreq, normalization)
   - `fnirs_unet.weights.h5`: U-Net weights
   - `feature_index.csv`: extracted features with quality tiers
   - `fnirs_synthetic_*.npz`: generated hemoglobin windows

### Multimodal Fusion
- Temporal cross-attention fusion on frame sequences across modalities.
- Each modality produces frame-level feature sequences; fusion attends across them.

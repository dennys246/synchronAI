## synchronAI agent notes

### Principles
- Keep changes minimal; reuse existing modules/folders when possible.
- Build so functionality and architecture can be used in unison or independently between multiple modalities (video/audio/fnirs)
- Creating larger systems is permissable to simplify handling of modalities however keep processes simple.
- Separate concerns: `data/` (I/O + preprocessing), `models/` (architectures), `training/` (loops), `inference/` (generation).
- fNIRS preprocessing must go through `hrfunc.preprocess_fnirs` (see `src/synchronai/data/fnirs/processing.py`).
- Prefer readable code over clever code: add short comments where intent isn’t obvious.
- Import dependencies as early as possible (module top) to reduce dependency mixups; if a deferred import is required (runtime config or optional dependency), isolate it in a helper with a short reason.
- Log progress to the terminal for long-running steps (training loops + diffusion sampling).
- Use Hugging Face repos as the canonical model distribution source; never commit access tokens or private artifacts.

### Hugging Face upload principles
- Repos: `dennys246/fNIRS_diffusion`, `dennys246/fNIRS_synchrony`, `dennys246/video_synchrony`, `dennys246/audio_synchrony`, `dennys246/transcript_synchrony`
- Always upload model artifacts from a clean `--save-dir` (config + weights + any metadata).
- Access tokens must be supplied via `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (or `--token` in CLI usage).
- Avoid uploading large caches or raw data; use allow/ignore patterns to keep the repo lean.

### Hugging Face process (upload + download)
- Upload:
  - `python -m synchronai.utils.hf_hub upload --model fnirs_diffusion --path runs/fnirs_diffusion`
- Download:
  - `python -m synchronai.utils.hf_hub download --model fnirs_diffusion --path runs/fnirs_diffusion`

### fNIRS diffusion workflow
- Train (expects a BIDS root or a single recording path):
  - `synchronai --fnirs --train diffusion --data-dir <PATH> --save-dir runs/fnirs_diffusion`
  - `python -m synchronai.main --modality fnirs --mode train --data-dir <PATH> --save-dir runs/fnirs_diffusion`
- Generate 120s synthetic hemoglobin windows from a trained checkpoint:
  - `synchronai --fnirs --generate diffusion --save-dir runs/fnirs_diffusion --n-samples 1`
  - `python -m synchronai.main --modality fnirs --mode generate --save-dir runs/fnirs_diffusion --n-samples 1`

Artifacts written under `--save-dir`:
- `fnirs_diffusion_config.json`: model/data metadata (pair names, sfreq, normalization)
- `fnirs_unet.weights.h5`: U-Net weights
- `fnirs_synthetic_*.npz`: generated hemoglobin windows

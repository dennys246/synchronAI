# Repository Reorganization Plan

This plan documents structural improvements to the synchronAI repo. Changes should be made once active training runs complete, except where noted.

---

## 1. Move Pretrained Weights Out of Repo Root

**Status:** Pending (waiting for runs to finish)

Move `yolo26s.pt` and `yolo26n-pose.pt` into a dedicated directory:

```
weights/
├── README.md              # expected files, download instructions
├── yolo26s.pt
└── yolo26n-pose.pt
```

**What to update after moving:**
- Any scripts or configs referencing `yolo26s.pt` / `yolo26n-pose.pt` by relative path
- `.gitignore` — add `weights/*.pt` if not already covered by `**/*.pt`

Optional: add a `scripts/download_weights.sh` that pulls weights from shared storage so collaborators don't copy manually.

---

## 2. Consolidate Documentation into docs/ (DONE)

**Status:** Complete (2026-03-23)

Moved all non-README docs from root and `src/` into `docs/`:

```
docs/
├── ARCHITECTURE.md
├── SIGNAL_TYPES.md
├── TRANSFER_LEARNING_FIXES.md
├── MULTIMODAL_HEATMAPS.md
├── audio_synchrony.md
├── transcript_synchrony.md
└── plans/
    ├── upgrade_plan.md
    ├── reorganization_plan.md
    └── fnirs_transfer_plan.md
```

Only `README.md` and `AGENTS.md` remain at root.

---

## 3. Standardize runs/ Directory Structure

**Status:** Pending (waiting for runs to finish)

### Naming convention
Use `{modality}_{model}_{experiment_tag}` consistently. Remove `_run_1` suffixes and `_old` copies.

### Standard run contents
Every run directory should contain:
```
runs/<experiment>/
├── config.yaml            # frozen snapshot of training config
├── best.pt (or .h5)       # best checkpoint
├── history.json            # training metrics
├── plots/                  # loss curves, confusion matrices
└── logs/                   # stdout/stderr from cluster jobs
```

### Cleanup
- Archive or delete `fnirs_diffusion_old/` if superseded
- Archive or delete `experiment/` template if unused
- Consider moving sweep results under a consistent `runs/sweeps/` prefix

---

## 4. Reorganize scripts/

**Status:** Pending

Current flat structure mixes training launchers, data utilities, and cluster jobs. Reorganize to:

```
scripts/
├── train/                 # shell scripts that invoke main.py
│   ├── train_audio_classifier.sh
│   ├── train_cv_synchrony.sh
│   ├── train_dinov2_features.sh
│   ├── train_dinov2_synchrony.sh
│   ├── train_multimodal_synchrony.sh
│   ├── generative_pretrain.sh
│   └── neural_pretrain.sh
├── cluster/               # bsub job scripts (renamed from bsub/)
│   ├── dinov2_sweep_bsub.sh
│   ├── multimodal_synchrony_bsub.sh
│   └── logs/
├── data/                  # dataset prep, feature extraction, auditing
│   ├── audit_dataset.py
│   ├── detect_persons.py
│   └── extract_dinov2_features.py
└── analysis/              # evaluation, profiling, comparison
    ├── compute_irr.py
    ├── compare_sweep_results.py
    └── profile_training.py
```

**What to update:** paths in bsub scripts that reference other scripts.

---

## 5. Organize data/ Directory

**Status:** Pending

```
data/
├── labels/                # all label CSVs
│   ├── labels.csv
│   ├── labels_train.csv
│   ├── labels_val.csv
│   ├── labels_train_mm_filtered.csv
│   └── labels_val_mm_filtered.csv
├── features/              # pre-extracted embeddings
│   ├── dinov2/
│   ├── dinov2_meanpatch/
│   ├── dinov2_small_meanpatch/
│   └── audio_sync/
└── raw/                   # sample videos, raw data (gitignored)
```

**What to update:**
- `configs/data/*.yaml` — label file and feature directory paths
- Any scripts with hardcoded `data/labels.csv` or `data/dinov2_features/` paths
- Consider generating train/val splits programmatically instead of storing separate CSVs

---

## 6. Add a Model Manifest

**Status:** Pending

Create a lightweight registry so finding the best checkpoint per modality doesn't require digging through `runs/`:

```yaml
# models/manifest.yaml
video:
  checkpoint: runs/dinov2_sweep/small_backbone/best.pt
  metric: val_acc=0.72
  date: 2026-03-10

audio:
  checkpoint: runs/audio_classifier/best.pt
  metric: val_acc=0.85
  date: 2026-03-08

fnirs:
  checkpoint: runs/fnirs_diffusion/fnirs_unet.weights.h5
  metric: val_loss=0.042
  date: 2026-03-05

multimodal:
  checkpoint: runs/multimodal_classifier/best.pt
  metric: val_acc=0.78
  date: 2026-03-12
```

Alternatively, add a script (`scripts/analysis/best_models.py`) that scans `runs/*/history.json` and reports the top result per experiment.

---

## 7. Move Cache Outside Repo

**Status:** Can do anytime

Set environment variables so HuggingFace cache doesn't land inside the repo:
```bash
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
```

Add to `.bashrc` / `.zshrc` or the project's `.envrc`.

---

## 8. Add Git Hash Tracking to Runs

**Status:** Pending (code change)

Have training scripts automatically record the git hash in saved configs:
```python
import subprocess
git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
config["git_hash"] = git_hash
```

This makes it easy to trace any run back to the exact code that produced it.

---

## Execution Order

Once active runs finish:

1. **Weights** — move `.pt` files to `weights/`, update references
2. **runs/** — standardize structure, clean up old/stale directories
3. **scripts/** — reorganize into subdirectories, update bsub paths
4. **data/** — reorganize into subdirectories, update configs and scripts
5. **Model manifest** — create after runs/ is clean
6. **Git hash tracking** — add to training code
7. **Cache** — set env vars, delete `.cache/` from repo

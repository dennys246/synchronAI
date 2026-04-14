# Troubleshooting Guide

Based on real problems encountered during synchronAI development.

---

## 1. Cluster / BSub Issues

### pip install race condition

- **Symptom**: `OSError: [Errno 2] No such file or directory: 'ml-env/bin/synchronai'` or `ModuleNotFoundError: No module named 'synchronai'`
- **Cause**: Multiple Docker containers mounting the same NFS filesystem run `pip install -e .` concurrently, corrupting the shared venv
- **Fix**: Don't use `pip install -e .` in BSub jobs. Set PYTHONPATH instead:
  ```bash
  export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
  ```
- **Prevention**: Console scripts entry point removed from pyproject.toml

### Heredoc variable expansion

- **Symptom**: Variables are empty inside BSub jobs, or paths have double slashes
- **Cause**: Unquoted heredocs (`<< EOF`) expand variables at submit time. Quoted heredocs (`<< 'EOF'`) expand at runtime.
- **Fix**: For unquoted heredocs, use `$VAR` directly (expands at submit). For quoted heredocs, variables must be in the environment (exported).
- **Rule of thumb**: Use unquoted heredocs when you need outer-scope variables (function args, computed paths). Use quoted heredocs when the job is self-contained and reads from env vars.

### Job dependency not working

- **Symptom**: Training job starts but feature_index.csv doesn't exist
- **Cause**: LSF `done()` fires when a job finishes, even if it failed (exit code 1). Training runs after a failed setup.
- **Fix**: Check setup logs for errors before relying on `done()` dependency. Setup scripts should `exit 1` on failure.

### NFS caching / file not found

- **Symptom**: File was just written by one job but another job can't find it
- **Cause**: NFS metadata caching across Docker containers
- **Fix**: Use absolute paths (`$SYNCHRONAI_DIR/data/...` not `data/...`). Add small delay if needed.

### SCRIPT_VERSION not showing in logs

- **Symptom**: Can't tell which script version ran
- **Cause**: Missing version echo in heredoc jobs
- **Fix**: Every BSub script and every heredoc must echo `=== [$SCRIPT_VERSION] ===` as first output

### Log file overwrites

- **Symptom**: Resubmitting a job overwrites the previous run's log
- **Cause**: LSF `-oo` with date-based filenames (`$DATE`) -- same date = same filename
- **Fix**: Check log timestamps. For important debugging, rename logs before resubmitting.

---

## 2. fNIRS Training Issues

### QC rejects all recordings

- **Symptom**: `RuntimeError: No valid windows found in any recording batch`
- **Cause**: QC thresholds too strict for the data (especially per-pair mode where individual pairs have lower SCI)
- **Fix**: Relax thresholds: `--sci-threshold 0.40 --snr-threshold 2.0 --no-require-cardiac`
- **Note**: The quality tier system (gold/standard/salvageable) handles quality filtering at training time

### Variance floor too high (std = 0.001)

- **Symptom**: `feature_std` in config is exactly `[0.001, 0.001]` regardless of data
- **Cause**: RunningStats variance floor was 1e-6, but per-pair HbO/HbR variance is ~1e-9
- **Fix**: Floor lowered to 1e-12 in `training/diffusion/train.py:RunningStats.std`
- **Impact**: Models trained with old floor are internally consistent (training + extraction use same std)

### Feature extraction produces no output

- **Symptom**: `feature_index.csv` missing after extraction "completes"
- **Cause**: All recordings QC-rejected, or encoder weights not found
- **Fix**: Check extraction logs for tier distribution. Verify encoder .pt exists.

### Holdout tiers empty

- **Symptom**: `Holdout tier 'salvageable': no val entries, skipping`
- **Cause**: Feature extraction didn't include salvageable tier, or no salvageable recordings in val split
- **Fix**: Extraction must use `--include-tiers "gold,standard,salvageable"` with relaxed QC thresholds

---

## 3. Video / Audio Training Issues

### DINOv2 OOM on extraction

- **Symptom**: CUDA out of memory during feature extraction
- **Fix**: Reduce batch size or use smaller resolution (112x112 -> 168 -> 224 progressive)

### WavLM content frames miscalculation

- **Symptom**: Audio features dominated by silence padding
- **Cause**: WavLM pads short audio to model length. Must pool only over content frames.
- **Fix**: `content_frames = max(1, int(total_frames * chunk_duration / 30.0))`

### Two-stage fine-tuning LR spike

- **Symptom**: Val loss jumps at stage 2 start
- **Cause**: Stage 2 head LR shouldn't reset to stage 1 LR. Use `backbone_lr * 5` instead.
- **Fix**: Warmup with `start_factor=0.3` for gentler stage 2 transition

---

## 4. Data Issues

### Subject ID mismatch across modalities

- **Symptom**: Feature join produces empty dataset
- **Cause**: Different ID formats across studies (CARE: 5-digit, P-CAT: with -C/-P suffix)
- **Fix**: Use `detect_participant_type()` in extract_fnirs_features.py for consistent labeling

### Label leakage across splits

- **Symptom**: Inflated val metrics that don't generalize
- **Cause**: Same subject's windows in both train and val
- **Fix**: Always use subject-grouped splitting (`group_by="subject_id"`)

# synchronAI — Claude Code Instructions

## Agent Principles (READ FIRST)

These apply to every Claude session and every agent spawned within one.

### Simplicity and correctness over ceremony

- **Write the simplest code that is correct.** Do not add abstractions, helpers, or
  wrappers unless the problem actually requires them. Three similar lines of code is
  better than a premature abstraction.
- **Do not add things that weren't asked for.** No bonus error handling, no extra
  logging, no "while I'm here" refactors, no docstrings on untouched code.
- **Small, focused changes.** If a task is narrow, the diff should be narrow. Resist
  the urge to clean up surrounding code.

### Honesty over sycophancy

- **Push back when appropriate.** If a proposed approach has a real flaw — wrong
  architecture, likely to break something, inconsistent with existing patterns — say
  so directly. Do not validate a bad idea just because the user seems committed to it.
- **Disagree with specifics, not vibes.** "That will cause an NFS race condition
  because X" is useful. "That seems risky" is not.
- **No empty affirmations.** Don't open responses with "Great idea!", "Perfect!",
  or similar. Get to the point.
- **Uncertainty is fine; false confidence is not.** If you don't know whether
  something will work, say so. Don't hedge everything, but don't overstate either.

### No band-aids

When fixing a bug or making a change, if you discover a deeper structural problem —
wrong abstraction, systemic misuse of an API, an issue that will recur — **stop and
surface it** before patching the symptom.

- If the underlying issue is minor, note it and ask whether to fix it now or log it.
- If the underlying issue is significant (architectural change, data corruption risk,
  would require touching many files), **pause the current task**, explain what you
  found and why it matters, and ask the user how to prioritize before writing any code.
- The bar for pausing is: "fixing this the narrow way will make the real problem
  harder to fix later." If that's true, the narrow fix is actively harmful.

Do not ship a workaround that obscures a real problem.

### Scope discipline

- If the task is ambiguous, ask one clarifying question rather than guessing broadly.
- If you notice a related problem while doing a task, note it briefly — don't fix it
  unilaterally unless it's a blocker.

---

## Development Workflow

### Branching

All development happens on focused feature branches off `main`. Do not work directly
on `main`.

- Branch naming: `feature/<short-description>` (e.g., `feature/fnirs-perpair-sweep`)
- One logical change per branch. If a session touches two unrelated things, split them.
- Branches should be **short-lived**: a session's worth of work, then push and open a PR.
  The human reviews and merges. Do not accumulate weeks of work on one branch.

### Pre-push checklist (MANDATORY)

Before pushing any branch, run a parallel architecture + execution analysis to
verify nothing is broken. Specifically:

1. **Architecture check**: Does the change alter any model input/output shapes,
   training interfaces, or data loading contracts that downstream code depends on?
   If yes, trace all callers.
2. **Execution flow check**: Does the change affect any BSub scripts, feature
   extraction paths, or weight file paths? If yes, verify versions were bumped and
   downstream jobs still resolve correctly.
3. **Import check**: Run `python -c "import synchronai"` (or equivalent) to verify
   the package still imports cleanly.

These checks should be done in parallel using subagents when the diff is non-trivial.
If a check surfaces a problem, fix it before pushing.

### Push cadence

- Push at the end of each work session, even if the branch isn't "done".
- A pushed-but-incomplete branch is better than local-only work. Cluster jobs pull
  from the repo — unpushed changes are invisible to them.
- Commit messages: one line summarizing *why*, not *what*. The diff shows what.

---

## BSub Script Versioning (MANDATORY)

Every BSub submission script in `scripts/bsub/` MUST include a version string that
gets echoed at the start of the job. This is critical for debugging cluster jobs —
logs must show which version of the script was actually executed.

### Rules

1. **Every BSub script** (`scripts/bsub/*.sh`) must define a `SCRIPT_VERSION` near the top
2. **Every heredoc job** must echo the version as its first output line
3. **Increment the version** any time the script logic changes (not just comments)
4. **Format**: `SCRIPT_VERSION="scriptname-vN"` where N is a monotonically increasing integer
5. **Echo format**: `echo "=== [SCRIPT_VERSION] ==="`

### Example — standalone BSub script

```bash
#!/bin/bash
#BSUB -G compute-perlmansusan
# ...
SCRIPT_VERSION="generative_fnirs_perpair-v3"
echo "=== [$SCRIPT_VERSION] ==="
```

### Example — launcher with heredoc jobs

```bash
SCRIPT_VERSION="pre_fnirs_child_adult_sweep-v5"
echo "=== [$SCRIPT_VERSION] ==="

bsub ... << EOF
echo "=== [$SCRIPT_VERSION] setup job ==="
# ... job body
EOF
```

### When to increment

- Any change to job logic, paths, flags, environment setup, or dependencies
- Do NOT increment for comment-only or whitespace changes
- When in doubt, increment

## Cluster Environment

- Shared filesystem: `/storage1/fs1/perlmansusan/Active/` mounted into Docker via `LSF_DOCKER_VOLUMES`
- Python env: `$SYNCHRONAI_DIR/ml-env/` (shared venv on NFS)
- **Do NOT use `pip install -e .`** in BSub jobs — causes race conditions when multiple
  Docker containers install into the same shared venv concurrently. Instead, set
  `PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"` to make the
  `synchronai` package importable.
- Use absolute paths (starting with `$SYNCHRONAI_DIR/`) for all feature dirs,
  save dirs, and weight paths. Relative paths fail inside Docker containers.
- Replace hardcoded usernames with `$USER` (e.g., `/home/$USER/.bashrc`)

### Filesystem Mount Points

The same NFS share is mounted at different paths depending on where you are:

| Location | Mount path |
|----------|-----------|
| **Cluster (BSub/Docker)** | `/storage1/fs1/perlmansusan/Active/moochie/` |
| **Mac (local dev)** | `/Volumes/perlmansusan/Active/moochie/` |

`$SYNCHRONAI_DIR` in BSub scripts must always use the cluster path
(`/storage1/fs1/...`). The Mac mount is read-only for checking logs and
results, but has aggressive NFS metadata caching — files written by
cluster jobs may not appear on the Mac for minutes to hours. Never rely
on the Mac mount to verify cluster job output; check logs instead.

Key implications:
- **Claude Code runs on the Mac mount** — file reads/edits go through
  `/Volumes/perlmansusan/Active/moochie/github/synchronAI/`
- **BSub jobs run on the cluster mount** — scripts must use
  `/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI`
- **Git repo is the same physical directory** — commits/pushes from
  either mount point affect the same repo
- **`data/` and `runs/` directories** may show stale content from the
  Mac due to NFS caching, but are up-to-date on the cluster

## fNIRS Pipeline

- All fNIRS work uses **per-pair architecture** (feature_dim=2, one HbO/HbR pair)
- The 20-channel format is reserved for higher-level models only
- Quality tiers (gold/standard/salvageable/rejected) are computed during feature
  extraction and stored in `quality_tier` column of `feature_index.csv`
- Training uses `--include-tiers "gold,standard"` for training data
- Holdout evaluation uses `--holdout-tiers "gold,salvageable"` for per-epoch
  monitoring of pristine vs high-motion data quality
- **QC caching (mandatory)**: All extraction jobs use `--qc-cache
  $SYNCHRONAI_DIR/data/qc_tiers.csv` — inline QC is ~50× slower and redundant
  once the cache exists (produced by `pre_fnirs_compute_qc_bsub.sh`)
- **Packed feature format (mandatory)**: Feature directories contain a single
  `features_packed.bin` + `features_meta.json`, loaded at training time via
  `np.memmap`. Individual `.pt` files are an extraction intermediate — never
  train on them. Always pass `--pack-output --delete-unpacked` to extraction.
  See [docs/fnirs_feature_storage.md](docs/fnirs_feature_storage.md).

## Naming Conventions

- **docs/ files**: lowercase with underscores (e.g., `fnirs_transfer_plan.md`)
- **Root-level files**: uppercase (e.g., `README.md`, `CLAUDE.md`, `AGENTS.md`) — standard convention
- **BSub scripts**: `{purpose}_{modality}_{variant}_bsub.sh` (e.g., `pre_fnirs_perpair_transfer_bsub.sh`)

## Code Conventions

- **Subject-grouped splits everywhere** — never use `random_split`. Group by `subject_id` to prevent data leakage.
- **Pre-extracted features to disk** for sweep training — extract once, train many configs.
- **Quality tiers** computed during extraction, filtered at training time via `--include-tiers`.
- **Two-stage fine-tuning** for video/audio: Stage 1 = frozen backbone, train head. Stage 2 = unfreeze with differential LRs (head LR = backbone_lr × 5, warmup start_factor=0.3).
- **Gradient clipping with AMP**: Always `scaler.unscale_(optimizer)` before `clip_grad_norm_`.
- **Label smoothing**: Manual cross-entropy with soft labels when using class weights.

## Common Cluster Issues

### pip install race condition
Multiple Docker containers running `pip install -e .` into the same shared NFS venv
corrupt the environment. **Fix**: Use `PYTHONPATH` instead of installing. See Cluster
Environment section above.

### Stale entry points
`ml-env/bin/synchronai` gets corrupted when multiple jobs install concurrently.
**Fix**: Removed `console_scripts` from `pyproject.toml`. Use `python -m synchronai.main`
instead of the `synchronai` CLI entry point.

### Heredoc variable expansion
Unquoted heredocs (`<< EOF`) expand variables at submit time. Quoted heredocs (`<< 'EOF'`)
expand at runtime. Both work if used consistently, but mixing them in the same script
causes empty variables. Pick one style per script and stick with it.

### `source` vs `.` in BSub scripts
LSF executes heredoc job bodies as POSIX `/bin/sh` scripts, not bash. `source` is a
bash built-in — it does not exist in `/bin/sh`. Use `.` instead:
```bash
# Wrong (fails silently in sh, never activates the venv):
source $SYNCHRONAI_DIR/ml-env/bin/activate
# Correct:
. $SYNCHRONAI_DIR/ml-env/bin/activate
```
Scripts with `#!/bin/bash` shebangs are fine. The failure mode is silent: `source: not
found` prints but the job continues with the system Python, not ml-env.

### GPFS slow `du` after mass deletion
The cluster filesystem is IBM Spectrum Scale (GPFS/rdcw-fs2). Directory hash tables
grow when files are created but **do not shrink when files are deleted**. Running `du`
on a directory that previously held 100K+ files is very slow even after deletion — GPFS
must traverse all the empty hash slots. This is not corruption. Use `mmdf` or
`mmlsquota` for quota/usage reporting instead of `du`.

### NFS caching
`feature_index.csv` may not be visible immediately after extraction completes on a
different node. **Fix**: Use absolute paths and, when chaining jobs, add a brief
verification step that checks the file exists before proceeding.

### Job dependency gotcha
`bsub -w "done(JOBID)"` fires on job completion regardless of exit code — a failed
setup job will still trigger downstream training. **Fix**: Use `"ended(JOBID)"` with
explicit exit-code checks at the start of dependent jobs, or use `"exit(JOBID, 0)"`
to only trigger on success.

## Refactoring Opportunities

These are known duplication/simplification targets for future cleanup:

- **TrainingHistory**: Duplicated across 4 training modules → extract shared base class
- **BinaryMetricTracker**: Duplicated in video + multimodal → extract to shared util
- **Feature extraction scripts**: All share discover → QC → extract → save pattern → common pipeline
- **BSub env setup**: conda init, PYTHONPATH, activate repeated ~28x → extract to `bsub_common.sh`

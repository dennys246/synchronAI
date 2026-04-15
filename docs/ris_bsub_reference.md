# WashU RIS BSub + Docker + ml-env reference

A reference for how cluster jobs run on WashU RIS's compute1 cluster in this
project, and the subtle gotchas we've hit. Intended for future you and future
Claude sessions — consult this before diagnosing cluster failures.

## The cluster in one picture

```
Login node (compute1-client-1.ris.wustl.edu)
│
│   — mounts NFS as /rdcw/fs2/perlmansusan/... AND /storage1/fs1/perlmansusan/...
│   — shell is bash, user home is /home/dennys
│   — enforces per-user quotas, address-space ulimits
│
└── bsub (submit jobs)
    │
    │   LSF scheduler dispatches to a compute node
    │
    ▼
Compute node (e.g. compute1-exec-411)
│
│   — starts a Docker container from the image specified in `-a docker(...)`
│   — mounts volumes listed in LSF_DOCKER_VOLUMES
│   — runs the heredoc body as /bin/sh
│
└── Docker container (continuumio/anaconda3)
    │
    │   — /opt/conda/bin/python (base anaconda python — NOT our project env)
    │   — /storage1/fs1/perlmansusan/Active/ mounted from NFS
    │   — no network access in general
    │
    └── The actual job body runs here
```

## The three filesystems

Same physical GPFS (IBM Spectrum Scale, mount name `rdcw-fs2`), three mount
points:

| Context | Path prefix |
|---|---|
| Login node | `/rdcw/fs2/perlmansusan/Active/moochie/` or `/storage1/fs1/perlmansusan/Active/moochie/` |
| BSub Docker container | `/storage1/fs1/perlmansusan/Active/moochie/` (via `LSF_DOCKER_VOLUMES`) |
| Local Mac (NFS mount) | `/Volumes/perlmansusan/Active/moochie/` |

**Always use `/storage1/fs1/...` inside BSub scripts.** That's the one both the
login node and the container can see at the same path. The `/rdcw/fs2/...`
path exists on the login node but **not inside the container**, which will
bite you if anything (venv, config, symlink) has that path hardcoded.

## The shared Python environment (`ml-env`)

Location: `$SYNCHRONAI_DIR/ml-env/`

- It's a regular Python venv, not a conda env. Created once, shared across
  all jobs via the NFS mount.
- Python: `ml-env/bin/python` → `python3.11` (symlink)
- site-packages: `ml-env/lib/python3.11/site-packages/` (and `lib64/` on some systems)
- Contains torch, tensorflow, transformers, CUDA libs, pandas, etc.

**Do NOT `pip install` into this venv from inside BSub jobs.** Multiple
concurrent containers writing to the same NFS site-packages is a race
condition that corrupts the env. If you need to add a dep, do it once,
manually, from a non-interactive login-node shell, and commit the
`pyproject.toml` change.

## The activation gotcha (documented 2026-04-14)

**`. "$SYNCHRONAI_DIR/ml-env/bin/activate"` inside LSF heredocs does not
reliably prepend `ml-env/bin` to `PATH`.** `python` keeps resolving to the
container's `/opt/conda/bin/python`, which doesn't have the project's
dependencies installed.

Symptom in logs:
```
--- diagnostic ---
which python: /opt/conda/bin/python        ← WRONG, should be ml-env
sys.executable: /opt/conda/bin/python
ModuleNotFoundError: No module named 'torch'
```

**Fix: invoke ml-env's python by absolute path.** Don't rely on activate.

```bash
# Inside a BSub heredoc:
cd $SYNCHRONAI_DIR
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:\$PYTHONPATH"
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"

"\$ML_PY" scripts/my_script.py --args ...
```

Python finds `ml-env/lib/python3.11/site-packages` automatically because
site-packages is resolved relative to the interpreter binary's location
— no PATH or activate needed. This is how all the fNIRS BSub scripts now
run as of v8/v9.

The pre-existing audio (wavlm) and video (dinov2) bsub scripts still use
`. activate` — they've empirically worked historically, possibly because
torch happens to be in the container's base anaconda python. If they start
failing with import errors, migrate them to the absolute-path pattern.

## Other BSub gotchas specific to this cluster

### Heredoc shell is /bin/sh, not bash

LSF writes heredoc bodies to `/home/$USER/.lsbatch/*.shell` and runs them
with `/bin/sh` (which is `dash` on this system, not bash). Consequences:

- `source` is bash-only — use `.` instead: `. /home/$USER/.bashrc`
- No `[[ ]]` — use `[ ]`
- No arrays
- No `function name()` syntax — use `name() { ... }`
- `echo -e` is not portable — use `printf` if you need escapes

The script containing the bsub command can use `#!/bin/bash` freely
(runs on the login node, not inside the heredoc).

### Heredoc variable expansion

**Unquoted** `<< EOF`: variables expand at submit time on the login node.
Use this when you want outer-scope values (function args, computed paths)
baked into the submitted job.

**Quoted** `<< 'EOF'`: variables expand inside the job at runtime.
Use this when the job is self-contained and reads only from job-scope env.

To mix: escape variables you want runtime-expanded with `\$VAR` inside an
unquoted heredoc. Example:
```bash
ML_PY="$SYNCHRONAI_DIR/ml-env/bin/python"   # submit-time, bakes absolute path
"\$ML_PY" scripts/my_script.py               # runtime, uses job's $ML_PY
```

### Job dependencies

- `-w "done(JOBID)"` fires only when the job exits with status 0
- `-w "ended(JOBID)"` fires on any exit (success or failure)
- `-w "exit(JOBID, 0)"` fires only on exit code 0 (same as done for most cases)

If a setup job fails, `done()`-dependent training jobs sit in `PEND`
state forever. `bkill -g /<group> 0` to clean them up.

### Docker image pulls

The first job on a new compute node pulls the Docker image (`continuumio/anaconda3`).
Subsequent jobs on the same node use the cached image. The pull lines in
logs (`Pulling fs layer`, `Pull complete`, `Digest: sha256:...`) are Docker,
not pip — they don't reflect your Python environment in any way.

### `LSF_DOCKER_VOLUMES` + `LSF_DOCKER_PRESERVE_ENVIRONMENT`

- `LSF_DOCKER_VOLUMES`: bind-mount host paths into the container. Required
  to get the shared NFS visible to the job. Must be set in the submitting
  shell *before* running `bsub`.
- `LSF_DOCKER_PRESERVE_ENVIRONMENT=true`: forwards environment variables
  from the submitting shell into the container. We rely on this for
  `SYNCHRONAI_DIR` and similar.

### Interactive vs non-interactive shells on the login node

- `pip install` often hits quota exceeded in an **interactive** login shell
  but succeeds in a **non-interactive** one. Exact cause unknown (likely
  PAM session limits or TMPDIR differences). Workaround: run installs
  non-interactively via `bash -c "..."` or a script.
- `du` on directories that previously held many files is **extremely slow**
  on GPFS because directory hash tables don't shrink on deletion. Use
  `mmdf` or `mmlsquota` for usage reporting instead.

### Script versioning

Every bsub script in `scripts/bsub/` has a `SCRIPT_VERSION="name-vN"` near
the top, echoed at the start of every job body. This is critical because
log filenames are date-based and cluster jobs may run across sessions —
the version string tells you which script version actually ran. Always
bump N on any logic change. See the main `CLAUDE.md` for full rules.

## Debugging checklist when a BSub job fails

1. **Find the log:** `grep -- "-oo" scripts/bsub/<script>.sh` shows you
   where the log goes. It's in `scripts/bsub/logs/` with a date suffix.
2. **Read the log.** Don't guess. The log has the actual error, the full
   job script (at the bottom), and the resource usage summary.
3. **Check `which python`** if you see any import error — are you actually
   running ml-env's python, or did you fall through to `/opt/conda/bin/python`?
4. **Check paths**: does the error mention `/rdcw/fs2/...`? If so, you
   have a path that's hardcoded to the login-node mount and doesn't exist
   in the container. Fix: use `/storage1/fs1/...` everywhere in bsub scripts.
5. **Check dependencies**: if a training job ran but features aren't there,
   the setup job probably failed but `done()` didn't catch it. Check the
   setup log (different filename) too.
6. **Check script version**: does the log's `=== [name-vN] ===` match the
   current source? If not, you're looking at an old run — re-run to get
   the latest.

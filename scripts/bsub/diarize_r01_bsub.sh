#!/bin/bash
SCRIPT_VERSION="diarize_r01-v5"
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 16000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 8
#BSUB -R 'select[mem>16GB && tmp>20GB] rusage[mem=16GB, tmp=20GB] span[hosts=1]'
#BSUB -J synchronai-diarize-r01
#BSUB -oo /storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/scripts/bsub/logs/diarize_r01_%J.log

# =============================================================================
# Run pyannote speaker diarization on P-CAT R01 DB-DOS recordings, then score it
# against the human verbal VAD.
#
# Why: docs/results/turn_structure_probe.md shows turn structure survives
# SIMULATED diarization error (0.709 -> 0.680 at ~28% DER), but that simulation
# injects i.i.d. per-second errors. Real errors are bursty. This measures the
# real DER and its run-length structure on records that already have human VAD.
#
# Two findings from that probe simplify the job:
#   * only SEPARATION matters, not identity -> num_speakers=2, no child/adult
#     classifier, no Voice Type Classification stage
#   * R01 "stereo" is duplicated mono (measured) -> the ffmpeg downmix is lossless
#
# Env vars:
#   DIAR_RECORDS : space-separated record IDs   (default: one smoke record)
#   DIAR_OUT     : output dir                   (default: data/diarization_r01)
#   HF_TOKEN     : optional. If unset, the token is read from HF_TOKEN_FILE.
#   HF_TOKEN_FILE: default secrets/.hf_token (git-ignored). pyannote's models are
#                  gated; accept the terms at
#                  huggingface.co/pyannote/speaker-diarization-3.1 and
#                  huggingface.co/pyannote/segmentation-3.0 first.
#                  The token is passed to python by path, never echoed.
#
# Deps live in a SEPARATE venv (diar-env3), not ml-env: pyannote pins torch and
# would break it, exactly as openSMILE did before prosodic-env was split out.
# CLAUDE.md bans pip install into the shared ml-env for that reason.
#
# v4 pins pyannote.audio <4. The 4.x SpeakerDiarization class loads the 3.1
# config and then pulls its own extra dependencies — a PLDA model from
# pyannote/speaker-diarization-community-1, which is access-restricted with no
# self-serve terms to accept, so it 403s and no amount of licence-clicking
# fixes it. The 3.x pipeline class has no PLDA step and matches the
# speaker-diarization-3.1 model card it was built for.
#
# New env PATH rather than deleting the old one: a batch job should not rm -rf a
# shared directory. Reclaim the 4.x env manually with `rm -rf diar-env`.
#
# v5 also pins torch/torchaudio. pyannote 3.x calls torchaudio.AudioMetaData,
# which newer torchaudio removed, so an unpinned install resolves to a pair
# 3.x predates and fails at import. The 2.2.x line is the newest that both
# supports Python 3.12 (the anaconda3 image ships 3.12, and torch <2.2 has no
# 3.12 wheels) and still exposes that attribute.
#
# The pip step now runs on EVERY invocation, not only when the venv is absent —
# otherwise a venv built with bad pins can never be corrected without deleting
# it by hand. pip is a no-op when the pins are already satisfied.
#
# Resources: pyannote-3.1 on CPU is ~30M params; peak is the model plus one
# decoded recording (~35 min at 16 kHz mono = 34 MB). 16GB is generous. Threads
# follow LSB_DJOB_NUMPROC so a -n override can't desync them.
# =============================================================================

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI"
cd "$SYNCHRONAI_DIR" || exit 1

DIAR_RECORDS="${DIAR_RECORDS:-11002}"
DIAR_OUT="${DIAR_OUT:-data/diarization_r01}"
VERBAL_CSV="${VERBAL_CSV:-verbal_pcat_r01_07-27-2026.csv}"
DIAR_ENV="${DIAR_ENV:-$SYNCHRONAI_DIR/diar-env3}"

# Per-job HOME on storage1: torch/pyannote write caches under $HOME, and the RIS
# home quota has crashed jobs (Errno 122) and hung them on NFS lock contention.
export HOME="$SYNCHRONAI_DIR/.jobhome/diarize_${LSB_JOBID:-local}"
mkdir -p "$HOME"
export HF_HOME="/storage1/fs1/perlmansusan/Active/moochie/resources/huggingface"
mkdir -p "$HF_HOME"
export PYTHONPATH="$SYNCHRONAI_DIR/src:$SYNCHRONAI_DIR:$PYTHONPATH"
NPROC="${LSB_DJOB_NUMPROC:-8}"
export OMP_NUM_THREADS="$NPROC"
export MKL_NUM_THREADS="$NPROC"

echo "=== [$SCRIPT_VERSION] ==="
echo "records=$DIAR_RECORDS"
echo "out=$DIAR_OUT  threads=$NPROC  HOME=$HOME"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$SYNCHRONAI_DIR/secrets/.hf_token}"
if [ -z "$HF_TOKEN" ] && [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: no HF token — \$HF_TOKEN is unset and $HF_TOKEN_FILE does not exist."
    echo "       pyannote's models are gated; accept the licences and provide a read token."
    exit 2
fi
if [ -n "$HF_TOKEN" ]; then
    echo "HF token: from \$HF_TOKEN"
else
    echo "HF token: from $HF_TOKEN_FILE"
fi

# --- one-time env bootstrap (serial: do NOT run two of these concurrently) ---
if [ ! -x "$DIAR_ENV/bin/python" ]; then
    echo "=== creating $DIAR_ENV (one-time) ==="
    python -m venv "$DIAR_ENV" || exit 1
    "$DIAR_ENV/bin/pip" install --quiet --upgrade pip || exit 1
fi
# PYTHONPATH is cleared for pip only: with the repo on it, pip sees synchronai
# and prints alarming "dependency conflict" ERROR lines about deps this env has
# no reason to hold. Pure noise, but it has already cost debugging time once.
echo "=== ensuring pinned deps ==="
env -u PYTHONPATH "$DIAR_ENV/bin/pip" install --quiet \
    "pyannote.audio>=3.1,<4" "torch>=2.2,<2.3" "torchaudio>=2.2,<2.3" || exit 1
DIAR_PY="$DIAR_ENV/bin/python"

"$DIAR_PY" - <<'PYCHK'
import sys
import pyannote.audio as pa, torch
import torchaudio
print("pyannote", pa.__version__, "torch", torch.__version__,
      "torchaudio", torchaudio.__version__)
if not hasattr(torchaudio, "AudioMetaData"):
    sys.exit("ERROR: torchaudio %s has no AudioMetaData; pyannote 3.x needs it. "
             "Pin torchaudio lower." % torchaudio.__version__)
major = int(pa.__version__.split(".")[0])
if major >= 4:
    sys.exit("ERROR: pyannote %s installed, but this pipeline needs 3.x — 4.x "
             "requires the access-restricted speaker-diarization-community-1 PLDA "
             "model. Remove %s and re-run to rebuild against the pin."
             % (pa.__version__, sys.prefix))
PYCHK
rc=$?
if [ $rc -ne 0 ]; then echo "ERROR: diar-env check failed (rc=$rc)"; exit $rc; fi

echo "=== [1/2] diarizing ==="
"$DIAR_PY" scripts/diarize_recordings.py --records $DIAR_RECORDS --out "$DIAR_OUT" \
    --scratch "${TMPDIR:-/tmp}" --token-file "$HF_TOKEN_FILE"
rc=$?
if [ $rc -ne 0 ]; then echo "ERROR: diarization failed (rc=$rc)"; exit $rc; fi

echo "=== [2/2] scoring against human VAD ==="
"$DIAR_PY" scripts/score_diarization.py --diar-dir "$DIAR_OUT" --verbal-csv "$VERBAL_CSV"
rc=$?
if [ $rc -ne 0 ]; then echo "ERROR: scoring failed (rc=$rc)"; exit $rc; fi

echo "=== [$SCRIPT_VERSION] complete ==="

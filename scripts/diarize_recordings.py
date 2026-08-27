#!/usr/bin/env python3
"""
Run speaker diarization on R01 DB-DOS recordings and emit per-second activity.

Purpose: replace the simulated DER curve in docs/results/turn_structure_probe.md
with a measurement. That curve injected i.i.d. per-second errors, which is an
optimistic model — real diarizer errors are bursty and systematic. Running an
actual diarizer against the 149 records that already carry human verbal VAD
gives both the real DER and its real error structure.

Two facts from that probe shape this script:
  * Speaker IDENTITY does not matter (swapping parent/child for whole records
    barely moved AUC), so pyannote's arbitrary SPEAKER_xx labels are fine and no
    child/adult classifier is needed.
  * Only SEPARATION matters, so num_speakers is pinned to 2 — this is closed-set
    diarization of a known dyad, not open-set clustering.

Output per record: <out>/<record_id>.npz with
    active : (n_seconds, 2) bool   - per-second activity for the two clusters
    seconds: int                   - length of the timeline
matching the per-second grid that extract_turn_features.py builds, so the two
can be scored against each other directly.

Requires pyannote.audio and a HF token with the gated models accepted:
    huggingface.co/pyannote/speaker-diarization-3.1
    huggingface.co/pyannote/segmentation-3.0

Usage:
    python scripts/diarize_recordings.py --records 11002 --out data/diarization_r01
    python scripts/diarize_recordings.py --records-file recs.txt --out data/diarization_r01
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOCAL_ROOT = "/Volumes/perlmansusan/Active/moochie"
CLUSTER_ROOT = "/storage1/fs1/perlmansusan/Active/moochie"
DATA_ROOT = next((r for r in (LOCAL_ROOT, CLUSTER_ROOT)
                  if os.path.isdir(f"{r}/study_data")), CLUSTER_ROOT)
VIDEO_ROOT = f"{DATA_ROOT}/study_data/P-CAT/R01/data/WUSTL_data/T1/video_data/dbdos"


def resolve_video(record_id: str) -> str | None:
    """Same anomaly handling as extract_turn_features.resolve_video: reject
    AppleDouble sidecars, partial-transfer artifacts, zero-byte remuxes and the
    loudness-fixed derivatives; accept the 'ddbos' typo; prefer the largest part."""
    d = Path(VIDEO_ROOT) / record_id
    if not d.is_dir():
        return None
    cands = []
    for p in d.iterdir():
        n = p.name
        if n.startswith("._") or "_fixed" in n:
            continue
        if not re.search(r"\.(mp4|mkv)$", n, re.IGNORECASE):
            continue
        if not re.search(r"d[db]dos", n, re.IGNORECASE):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz:
            cands.append((sz, p))
    return str(max(cands)[1]) if cands else None


def to_wav(video: str, wav: Path, sr: int = 16000) -> None:
    """Decode to 16 kHz mono. R01 'stereo' is duplicated mono (measured), so the
    downmix is lossless here, not a compromise."""
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video,
           "-map", "0:a:0", "-ac", "1", "-ar", str(sr), str(wav)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not wav.exists():
        raise RuntimeError(f"ffmpeg failed for {video}: {r.stderr.strip()[:300]}")


def diarize(wav: Path, pipeline) -> tuple[np.ndarray, int]:
    ann = pipeline({"uri": wav.stem, "audio": str(wav)}, num_speakers=2)
    segs = [(s.start, s.end, spk) for s, _, spk in ann.itertracks(yield_label=True)]
    if not segs:
        return np.zeros((0, 2), dtype=bool), 0
    n = int(np.ceil(max(e for _, e, _ in segs)))
    labels = sorted({spk for *_, spk in segs})[:2]
    out = np.zeros((n, 2), dtype=bool)
    for st, en, spk in segs:
        if spk not in labels:
            continue
        j = labels.index(spk)
        out[int(np.floor(st)):int(np.ceil(en)), j] = True
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--records", nargs="+", help="Record IDs, e.g. 11002 11004")
    g.add_argument("--records-file", help="File with one record ID per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    args = ap.parse_args()

    recs = args.records or [l.strip() for l in open(args.records_file) if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "ERROR: HF_TOKEN not set. pyannote's models are gated — accept the terms at\n"
            "  huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  huggingface.co/pyannote/segmentation-3.0\n"
            "then export a read token as HF_TOKEN.")

    from pyannote.audio import Pipeline
    logger.info("Loading %s (CPU)", args.model)
    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=token)

    n_ok = n_skip = n_fail = 0
    for rid in recs:
        dst = out / f"{rid}.npz"
        if dst.exists():
            logger.info("%s: exists, skipping", rid)
            n_skip += 1
            continue
        vid = resolve_video(rid)
        if vid is None:
            logger.warning("%s: no resolvable recording", rid)
            n_fail += 1
            continue
        wav = Path(args.scratch) / f"{rid}.wav"
        try:
            to_wav(vid, wav)
            active, n = diarize(wav, pipeline)
            if n == 0:
                logger.warning("%s: diarizer returned no segments", rid)
                n_fail += 1
                continue
            np.savez_compressed(dst, active=active, seconds=n, source=vid)
            logger.info("%s: %d s, cluster activity %.3f / %.3f", rid, n,
                        active[:, 0].mean(), active[:, 1].mean())
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            logger.error("%s: FAILED %s", rid, e)
            n_fail += 1
        finally:
            if wav.exists():
                wav.unlink()

    logger.info("done: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail)
    return 1 if n_fail and not n_ok else 0


if __name__ == "__main__":
    sys.exit(main())

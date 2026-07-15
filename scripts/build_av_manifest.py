#!/usr/bin/env python3
"""Build a per-recording video/audio manifest CSV.

Walks the configured video/audio study roots, classifies each file via the
shared `synchronai.data.metadata.classify_recording()`, probes per-file
attributes (duration, codec, fps, resolution, sample rate, channels) via
ffprobe, and writes a single `av_manifest.csv`.

This is the video/audio analogue of `data/qc_tiers.csv` + feature_index.csv
for fNIRS — a canonical list of what recordings exist, from which study,
at which timepoint, for which subject.

Usage:
    python scripts/build_av_manifest.py
    python scripts/build_av_manifest.py --output data/av_manifest.csv
    python scripts/build_av_manifest.py --no-probe     # skip ffprobe, faster
    python scripts/build_av_manifest.py --data-dirs "rootA:rootB"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Ensure src/ is importable when this script is run directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from synchronai.data.metadata import (  # noqa: E402
    AUDIO_EXTS,
    MODALITY_AUDIO,
    MODALITY_VIDEO,
    VIDEO_EXTS,
    classify_recording,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Default roots. Uses cluster paths; override with --data-dirs.
# Only studies with on-disk video/audio are listed — R01 is fNIRS-only today.
DEFAULT_AV_ROOTS = [
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/video_data/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/CARE/audio_data/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/video_data/",
    "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/audio_data/",
]


def discover_av_files(data_dirs: list[str]) -> list[tuple[str, str]]:
    """Return [(path, modality)] for every video/audio file under the roots."""
    out: list[tuple[str, str]] = []
    for root in data_dirs:
        root = root.strip()
        if not root:
            continue
        if not Path(root).exists():
            logger.warning("Skipping missing directory: %s", root)
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                # Skip macOS AppleDouble sidecars (./._filename) created when
                # writing to non-HFS filesystems.
                if fname.startswith("._"):
                    continue
                ext = Path(fname).suffix.lower()
                if ext in VIDEO_EXTS:
                    out.append((str(Path(dirpath) / fname), MODALITY_VIDEO))
                elif ext in AUDIO_EXTS:
                    out.append((str(Path(dirpath) / fname), MODALITY_AUDIO))
    out.sort(key=lambda x: x[0])
    return out


_PROBE_BACKEND_CHECKED = False
_FFPROBE_AVAILABLE = False
_CV2 = None
_MUTAGEN = None


def _init_probe_backends() -> None:
    """Detect which probing backends are available and log once.

    Preference order: ffprobe > cv2 (video) / mutagen (audio). Any backend
    being unavailable just means that modality's fields stay empty.
    """
    global _PROBE_BACKEND_CHECKED, _FFPROBE_AVAILABLE, _CV2, _MUTAGEN
    if _PROBE_BACKEND_CHECKED:
        return
    _PROBE_BACKEND_CHECKED = True

    _FFPROBE_AVAILABLE = shutil.which("ffprobe") is not None

    try:
        import cv2 as _cv2_mod
        _CV2 = _cv2_mod
    except ImportError:
        _CV2 = None

    try:
        import mutagen as _mutagen_mod
        _MUTAGEN = _mutagen_mod
    except ImportError:
        _MUTAGEN = None

    backends = []
    if _FFPROBE_AVAILABLE:
        backends.append("ffprobe")
    if _CV2 is not None:
        backends.append("cv2(video)")
    if _MUTAGEN is not None:
        backends.append("mutagen(audio)")
    if backends:
        logger.info("Probe backends available: %s", ", ".join(backends))
    else:
        logger.warning(
            "No probe backends available (ffprobe/cv2/mutagen all missing) — "
            "duration/codec columns will be empty. "
            "Install any one of: ffmpeg, opencv-python, mutagen. "
            "Or pass --no-probe to silence this."
        )


def _probe_with_cv2(path: str) -> dict:
    """Video probe via OpenCV. Reads header only; works for common codecs."""
    if _CV2 is None:
        return {}
    try:
        cap = _CV2.VideoCapture(path)
        if not cap.isOpened():
            return {}
        try:
            fps = cap.get(_CV2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(_CV2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(_CV2.CAP_PROP_FRAME_HEIGHT) or 0)
            out: dict = {}
            if fps > 0 and frame_count > 0:
                out["duration_seconds"] = frame_count / fps
                out["video_fps"] = fps
            if width:
                out["video_width"] = width
            if height:
                out["video_height"] = height
            return out
        finally:
            cap.release()
    except Exception:
        return {}


def _probe_with_mutagen(path: str) -> dict:
    """Audio probe via mutagen. Reads headers only — fast, no decode."""
    if _MUTAGEN is None:
        return {}
    try:
        f = _MUTAGEN.File(path)
        if f is None or f.info is None:
            return {}
        out: dict = {}
        if getattr(f.info, "length", None):
            out["duration_seconds"] = float(f.info.length)
        if getattr(f.info, "sample_rate", None):
            out["audio_sample_rate"] = int(f.info.sample_rate)
        if getattr(f.info, "channels", None):
            out["audio_channels"] = int(f.info.channels)
        return out
    except Exception:
        return {}


def probe_file(path: str) -> dict:
    """Probe a media file for duration/codec/etc.

    Tries ffprobe first, then falls back to cv2 (video) or mutagen (audio).
    Returns an empty dict if no backend succeeds. Callers should treat every
    field as optional.
    """
    _init_probe_backends()

    if _FFPROBE_AVAILABLE:
        out = _probe_with_ffprobe(path)
        if out:
            return out

    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return _probe_with_cv2(path)
    if ext in AUDIO_EXTS:
        return _probe_with_mutagen(path)
    return {}


def _probe_with_ffprobe(path: str) -> dict:
    if not _FFPROBE_AVAILABLE:
        return {}
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_format", "-show_streams",
                "-of", "json",
                path,
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return {}

    out: dict = {}
    fmt = data.get("format", {}) or {}
    duration = fmt.get("duration")
    if duration is not None:
        try:
            out["duration_seconds"] = float(duration)
        except (TypeError, ValueError):
            pass
    size = fmt.get("size")
    if size is not None:
        try:
            out["file_size_bytes"] = int(size)
        except (TypeError, ValueError):
            pass

    for stream in data.get("streams", []) or []:
        kind = stream.get("codec_type")
        if kind == "video" and "video_codec" not in out:
            out["video_codec"] = stream.get("codec_name", "")
            out["video_width"] = stream.get("width")
            out["video_height"] = stream.get("height")
            fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            if fps and fps != "0/0":
                try:
                    num, den = fps.split("/")
                    out["video_fps"] = float(num) / float(den) if float(den) else None
                except (ValueError, ZeroDivisionError):
                    pass
        elif kind == "audio" and "audio_codec" not in out:
            out["audio_codec"] = stream.get("codec_name", "")
            sr = stream.get("sample_rate")
            if sr is not None:
                try:
                    out["audio_sample_rate"] = int(sr)
                except (TypeError, ValueError):
                    pass
            ch = stream.get("channels")
            if ch is not None:
                try:
                    out["audio_channels"] = int(ch)
                except (TypeError, ValueError):
                    pass
    return out


# Schema: manifest column order. Extras from classifier + probe are appended.
_BASE_COLUMNS = [
    "recording_id", "session_id",
    "modality", "study", "site", "timepoint", "task",
    "family_id", "subject_id", "participant_type",
    "source_path",
]


def build_manifest(paths: list[tuple[str, str]], probe: bool) -> pd.DataFrame:
    rows: list[dict] = []
    iterator = tqdm(paths, desc="Classifying" + (" + probing" if probe else ""))
    for path, modality in iterator:
        meta = classify_recording(path, modality=modality)
        row = meta.to_dict()
        if probe:
            row.update(probe_file(path))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_BASE_COLUMNS)

    df = pd.DataFrame(rows)
    # Stable column order: base columns first, then anything else (probe + extras).
    base = [c for c in _BASE_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in base]
    return df[base + rest]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dirs", default=None,
        help="Colon-separated video/audio roots. Defaults to CARE + R56 cluster paths.",
    )
    parser.add_argument(
        "--output", default="data/av_manifest.csv",
        help="Where to write the manifest CSV.",
    )
    parser.add_argument(
        "--no-probe", action="store_true",
        help="Skip ffprobe. Produces rows without duration/codec/etc.",
    )
    args = parser.parse_args()

    roots = args.data_dirs.split(":") if args.data_dirs else list(DEFAULT_AV_ROOTS)
    logger.info("Scanning %d AV roots", len(roots))

    paths = discover_av_files(roots)
    logger.info("Discovered %d video/audio files", len(paths))
    if not paths:
        logger.error("No files found. Check --data-dirs.")
        return 1

    df = build_manifest(paths, probe=not args.no_probe)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)

    # Quick sanity report so the user sees coverage at a glance.
    n_unknown_study = int((df["study"] == "unknown").sum())
    n_unknown_pt = int((df["participant_type"] == "unknown").sum())
    print()
    print(f"By modality:  {df['modality'].value_counts().to_dict()}")
    print(f"By study:     {df['study'].value_counts().to_dict()}")
    if n_unknown_study:
        print(f"WARNING: {n_unknown_study} files with unclassified study.")
    if n_unknown_pt:
        print(f"WARNING: {n_unknown_pt} files with unknown participant_type.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

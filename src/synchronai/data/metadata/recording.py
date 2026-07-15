"""Recording-level metadata: canonical schema + classification dispatcher.

This module is the single source of truth for mapping a raw-data path
(fNIRS directory, video file, audio file) to a `RecordingMetadata` record
with stable study / timepoint / task / subject fields.

Adding a new study: add a classifier in `studies.py` and register it in
`_STUDY_CLASSIFIERS` below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

UNKNOWN = "unknown"

MODALITY_FNIRS = "fnirs"
MODALITY_VIDEO = "video"
MODALITY_AUDIO = "audio"

VIDEO_EXTS = {".mp4", ".mov", ".mts", ".m4v", ".avi", ".mkv"}
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg"}


@dataclass
class RecordingMetadata:
    """Canonical recording metadata. One row per raw file (or NIRx dir).

    Field semantics:
        modality:         {fnirs, video, audio, unknown}
        study:            {CARE, R56, R01_PSU, R01_WUSTL, unknown}
        site:             {WashU, PSU, WUSTL, unknown} — only meaningful for R01
        timepoint:        study-native code (V0/V1/V2, T1/T3/T5, visit1, unknown)
        task:             DB-DOS / Flanker / Interview / TSST / ..., unknown
        family_id:        family/dyad identifier (e.g. "5000", "1102", "11001")
        subject_id:       stable per-participant ID across visits
                          (CARE: "50001"; R01: "11001_C"; R56: "1102-C";
                          for dyad-level recordings: "{family}-dyad" or
                          similar — participant_type == "dyad")
        participant_type: {child, adult, dyad, unknown}
        source_path:      absolute path to the file or NIRx directory
        session_id:       cross-modal join key
                          {study}__{timepoint}__{subject_id}__{task}
        recording_id:     {session_id}__{modality}; unique per file
        extras:           optional per-modality fields (camera_angle,
                          file_format, etc.) — never relied on upstream
    """

    source_path: str
    modality: str = UNKNOWN
    study: str = UNKNOWN
    site: str = UNKNOWN
    timepoint: str = UNKNOWN
    task: str = UNKNOWN
    family_id: str = ""
    subject_id: str = ""
    participant_type: str = UNKNOWN
    session_id: str = ""
    recording_id: str = ""
    extras: dict = field(default_factory=dict)

    def finalize(self) -> "RecordingMetadata":
        """Compute derived IDs. Call after all study/subject fields are set."""
        if not self.session_id:
            self.session_id = "__".join([
                self.study or UNKNOWN,
                self.timepoint or UNKNOWN,
                self.subject_id or UNKNOWN,
                self.task or UNKNOWN,
            ])
        if not self.recording_id:
            self.recording_id = f"{self.session_id}__{self.modality or UNKNOWN}"
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten extras as-is for CSV writers; leave callers to merge if needed.
        extras = d.pop("extras", {}) or {}
        d.update(extras)
        return d


def infer_modality(path: str | Path) -> str:
    """Infer modality from a path or filename.

    fNIRS detection is intentionally permissive because NIRx recordings are
    directories (no extension) and SNIRF/FIF files also count.
    """
    p = Path(path)
    parts_lower = {x.lower() for x in p.parts}

    if "nirs_data" in parts_lower or "fnirs" in parts_lower:
        return MODALITY_FNIRS

    ext = p.suffix.lower()
    if ext in VIDEO_EXTS:
        return MODALITY_VIDEO
    if ext in AUDIO_EXTS:
        return MODALITY_AUDIO

    # Directories without an extension — assume fNIRS NIRx dir if it looks like one.
    if ext == "" and ("video_data" not in parts_lower and "audio_data" not in parts_lower):
        return MODALITY_FNIRS

    if "video_data" in parts_lower:
        return MODALITY_VIDEO
    if "audio_data" in parts_lower:
        return MODALITY_AUDIO

    return UNKNOWN


def detect_study(path: str | Path) -> str:
    """Return the study key for a path, or 'unknown'.

    Study keys match those used by the fNIRS pipeline for backward compatibility:
    CARE, R56, R01_PSU, R01_WUSTL.
    """
    parts = set(Path(path).parts)
    if "CARE" in parts:
        return "CARE"
    if "R56" in parts:
        return "R56"
    # R01 is split by site.
    if "PSU_data" in parts:
        return "R01_PSU"
    if "WUSTL_data" in parts:
        return "R01_WUSTL"
    return UNKNOWN


# Registered in studies.py to avoid a circular import.
_STUDY_CLASSIFIERS: dict[str, Callable[[str, str, RecordingMetadata], RecordingMetadata]] = {}


def register_study_classifier(
    study: str,
    fn: Callable[[str, str, RecordingMetadata], RecordingMetadata],
) -> None:
    """Register a `(path, modality, meta) -> meta` classifier for a study."""
    _STUDY_CLASSIFIERS[study] = fn


def classify_recording(path: str, modality: Optional[str] = None) -> RecordingMetadata:
    """Classify a raw-data path into a RecordingMetadata record.

    `modality` may be passed explicitly (useful when the caller already knows,
    e.g., during an audio-only walk). Otherwise it is inferred from the path.
    """
    if modality is None:
        modality = infer_modality(path)

    meta = RecordingMetadata(source_path=str(path), modality=modality)
    meta.study = detect_study(path)

    classifier = _STUDY_CLASSIFIERS.get(meta.study)
    if classifier is not None:
        meta = classifier(str(path), modality, meta)

    return meta.finalize()


# Import studies to trigger classifier registration. This runs the side-effectful
# register_study_classifier calls in studies.py.
from . import studies as _studies  # noqa: E402, F401

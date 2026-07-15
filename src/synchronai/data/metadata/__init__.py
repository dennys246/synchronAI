"""Shared recording-level metadata schema + classifier for synchronAI.

Use `classify_recording(path)` to map a raw-data path (fNIRS NIRx dir, SNIRF/
FIF file, video/audio file) to a `RecordingMetadata` record with canonical
study / timepoint / task / subject fields.

See `recording.py` for the schema, `studies.py` for per-study parsers.
"""

from .recording import (
    AUDIO_EXTS,
    MODALITY_AUDIO,
    MODALITY_FNIRS,
    MODALITY_VIDEO,
    RecordingMetadata,
    UNKNOWN,
    VIDEO_EXTS,
    classify_recording,
    detect_study,
    infer_modality,
    register_study_classifier,
)

__all__ = [
    "AUDIO_EXTS",
    "MODALITY_AUDIO",
    "MODALITY_FNIRS",
    "MODALITY_VIDEO",
    "RecordingMetadata",
    "UNKNOWN",
    "VIDEO_EXTS",
    "classify_recording",
    "detect_study",
    "infer_modality",
    "register_study_classifier",
]

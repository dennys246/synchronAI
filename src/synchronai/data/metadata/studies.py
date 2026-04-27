"""Per-study path classifiers.

Each classifier takes `(path, modality, meta)` where `meta.study` is already
set, and returns `meta` with study-specific fields filled in. Unrecognized
paths should leave the default 'unknown' fields in place rather than guessing.

Path patterns (verified against the on-disk layout, 2026-04):

CARE
  fNIRS (dir):  .../CARE/NIRS_data/{family4}/{V0|V1|V2}/{subject}_V{N}_fNIRS/
  video (file): .../CARE/video_data/{family4}/{V0|V1|V2}/{subject}_V{N}_{task}.mp4
  audio (file): .../CARE/audio_data/{family4}/{V0|V1|V2}/{subject}_V{N}_{task}_audio.m4a
  Participant: 4-digit subject_id = adult; 5-digit = child (shares family prefix).

R56 (single-visit: "visit1")
  fNIRS (dir):  .../R56/NIRS_data/{family4}/{family}-{C,P}_fNIRS_{task}/   # per-participant
                .../R56/NIRS_data/{family4}/{family}_{task}/                # dyad-level
  video (file): .../R56/video_data/{family4}/{family}_{task}.mp4            # dyad-level
                .../R56/video_data/{family4}/{family}_TSST/{family}_TSST_{angle}.{mp4,MTS}
  audio (file): .../R56/audio_data/{family4}/{family}_{task}_audio.m4a      # dyad-level

R01 PSU / WUSTL  (fNIRS only on disk at time of writing)
  fNIRS (dir):  .../R01/data/PSU_share/{site}_data/{T1|T3|T5}/nirs_data/dbdos/{family}/{subject}/
  Participant: subject_id is `{family}_C` (child) or `{family}_P` (adult).
"""

from __future__ import annotations

import re
from pathlib import Path

from .recording import (
    MODALITY_AUDIO,
    MODALITY_FNIRS,
    MODALITY_VIDEO,
    RecordingMetadata,
    register_study_classifier,
)


# Canonical task names. Filenames may use any case, separator, or annotation;
# we normalize and look up so "interview", "Interview", "Interview_Original",
# "DB_DOS", "DB-DOS (extra)" all collapse to the canonical form.
_TASK_CANON = {
    "db-dos": "DB-DOS",
    "db-d0s": "DB-DOS",       # Known source typo (zero vs letter O).
    "flanker": "Flanker",
    "interview": "Interview",
    "tsst": "TSST",
    "arts&crafts": "Arts&Crafts",
    "puzzles": "Puzzles",
    "magnettiles": "MagnetTiles",
    "jumble": "Jumble",       # Seen in R56 noldus sample videos.
}

# Duplicate-take / version markers that appear as a suffix on an otherwise
# canonical task, e.g. "Interview_Original", "DB-DOS_Final".
_DUP_MARKER_SUFFIXES = ("_original", "_final", "_complete", "_inprogress")

# Task keywords to scan for when the filename is free-form (noldus samples).
# Ordered so longer/more-specific keywords win before shorter ones.
_TASK_KEYWORDS = [
    ("DB-DOS", "DB-DOS"),
    ("DB_DOS", "DB-DOS"),
    ("FLANKER", "Flanker"),
    ("INTERVIEW", "Interview"),
    ("MAGNETTILES", "MagnetTiles"),
    ("ARTS", "Arts&Crafts"),
    ("PUZZLES", "Puzzles"),
    ("JUMBLE", "Jumble"),
    ("TSST", "TSST"),
]


def _canonicalize_task(task: str) -> str:
    """Collapse case/separator/annotation variants to a canonical task name.

    Returns the canonical form if known; otherwise returns the cleaned input
    so consumers can still see what the source file said.
    """
    if not task:
        return task
    # Strip trailing annotations like " (extra)".
    task = re.sub(r"\s*\([^)]*\)\s*$", "", task).strip()
    # Strip known duplicate-take suffixes.
    lowered = task.lower()
    for suffix in _DUP_MARKER_SUFFIXES:
        if lowered.endswith(suffix):
            task = task[: -len(suffix)]
            lowered = task.lower()
            break
    # Normalize underscore separator to hyphen for lookup ("DB_DOS" -> "db-dos").
    lookup = lowered.replace("_", "-")
    if lookup in _TASK_CANON:
        return _TASK_CANON[lookup]
    return _TASK_CANON.get(lowered, task)


def _task_from_freeform(text: str) -> str:
    """Scan a free-form filename for known task keywords."""
    upper = text.upper()
    for keyword, canonical in _TASK_KEYWORDS:
        if keyword in upper:
            return canonical
    return "unknown"


# Filename stem patterns used by video/audio classifiers.
#
# The `task` group uses `.+?` (lazy) so it accepts separators like `_`, `-`,
# spaces, and annotations like "(extra)". `_canonicalize_task` then collapses
# those variants to a canonical form.
#
# Handled variants (both modalities):
#   {subject}_V{N}_{task}                      e.g. 50001_V0_DB-DOS
#   {subject}_V{N}_{task}_audio                e.g. 5000_V0_Interview_audio
#   {subject}_V{N}_{task}_audio_{part}         e.g. 5003_V0_Interview_audio_1
#   {subject}_{task}_audio_{part}              e.g. 5035_Interview_audio_1
#                                              (timepoint recovered from parent dir)
#   {subject}_V{N}_DB_DOS                      e.g. 51891_V2_DB_DOS
#   {family}_{task}_Audio                      e.g. 1144_Interview_Audio
#                                              (case-insensitive audio suffix)
#   {family}_{task} (extra)                    e.g. 1126_DB-DOS (extra)
#   {family}_{task}_Original                   e.g. 1125_Interview_Original
_CARE_AV_STEM = re.compile(
    r"^(?P<subject>\d{4,5})"
    r"(?:_V(?P<tp>\d+))?"
    r"_(?P<task>.+?)"
    r"(?:_[Aa]udio)?"
    r"(?:_(?P<part>\d+))?"
    r"$"
)
_R56_AV_STEM = re.compile(
    r"^(?P<family>\d{4})"
    r"_(?P<task>.+?)"
    r"(?:_[Aa]udio)?"
    r"(?:_(?P<part>\d+))?"
    r"$"
)
_R56_AV_TSST_STEM = re.compile(
    r"^(?P<family>\d{4})_TSST_(?P<angle>[A-Za-z]+)(?:_[Aa]udio)?(?:_(?P<part>\d+))?$"
)


def _classify_care(path: str, modality: str, meta: RecordingMetadata) -> RecordingMetadata:
    meta.site = "WashU"
    parts = Path(path).parts
    stem = Path(path).stem

    if modality == MODALITY_FNIRS:
        # DB-DOS is the only fNIRS task captured in CARE.
        meta.task = "DB-DOS"
        for p in parts:
            if re.fullmatch(r"V\d+", p):
                meta.timepoint = p
                break
        for p in parts:
            m = re.match(r"^(\d+)_V\d+_fNIRS", p)
            if m:
                sub = m.group(1)
                meta.subject_id = sub
                meta.family_id = sub[:4] if len(sub) >= 4 else sub
                if len(sub) == 5:
                    meta.participant_type = "child"
                elif len(sub) == 4:
                    meta.participant_type = "adult"
                break
        return meta

    if modality in (MODALITY_VIDEO, MODALITY_AUDIO):
        # Timepoint comes from the parent directory.
        for p in parts:
            if re.fullmatch(r"V\d+", p):
                meta.timepoint = p
                break
        m = _CARE_AV_STEM.match(stem)
        if m:
            sub = m.group("subject")
            meta.subject_id = sub
            meta.family_id = sub[:4] if len(sub) >= 4 else sub
            meta.task = _canonicalize_task(m.group("task"))
            if not meta.timepoint and m.group("tp") is not None:
                meta.timepoint = f"V{m.group('tp')}"
            if m.group("part") is not None:
                meta.extras["part"] = m.group("part")
            # CARE A/V files are typically dyadic (both in frame). We encode
            # participant_type from ID length for parity with fNIRS, but note
            # the recording contains both people — consumers that need a
            # dyad-level key should use family_id.
            if len(sub) == 5:
                meta.participant_type = "child"
            elif len(sub) == 4:
                meta.participant_type = "adult"
        meta.extras["file_ext"] = Path(path).suffix.lower().lstrip(".")
        return meta

    return meta


def _classify_r56(path: str, modality: str, meta: RecordingMetadata) -> RecordingMetadata:
    meta.site = "WashU"
    meta.timepoint = "visit1"  # R56 is single-visit.
    parts = Path(path).parts
    stem = Path(path).stem

    if modality == MODALITY_FNIRS:
        # Task detection from path (directory name holds it).
        path_upper = path.upper()
        if "FLANKER" in path_upper:
            meta.task = "Flanker"
        else:
            meta.task = "DB-DOS"

        # Per-participant dir: "1102-C_fNIRS_DB-DOS"
        for p in parts:
            m = re.match(r"^(\d+)-([CP])_fNIRS", p)
            if m:
                family, ct = m.group(1), m.group(2)
                meta.family_id = family
                meta.subject_id = f"{family}-{ct}"
                meta.participant_type = "child" if ct == "C" else "adult"
                return meta

        # Family-level Flanker dir: "1109_Flanker"
        for p in parts:
            m = re.match(r"^(\d+)_Flanker", p)
            if m:
                family = m.group(1)
                meta.family_id = family
                meta.subject_id = f"{family}-dyad"
                meta.participant_type = "dyad"
                return meta
        return meta

    if modality in (MODALITY_VIDEO, MODALITY_AUDIO):
        # The `noldus/` folder contains calibration and demo clips from the
        # Noldus Observer recording suite — not participant data. Tag as
        # samples so downstream filters can exclude them.
        if "noldus" in [p.lower() for p in parts]:
            meta.participant_type = "sample"
            meta.task = _task_from_freeform(stem)
            meta.extras["is_sample"] = True
            meta.extras["file_ext"] = Path(path).suffix.lower().lstrip(".")
            return meta

        # TSST has a sub-folder with multiple camera angles.
        m_tsst = _R56_AV_TSST_STEM.match(stem)
        if m_tsst:
            family = m_tsst.group("family")
            meta.family_id = family
            meta.subject_id = f"{family}-dyad"
            meta.participant_type = "dyad"
            meta.task = "TSST"
            meta.extras["camera_angle"] = m_tsst.group("angle")
            if m_tsst.group("part") is not None:
                meta.extras["part"] = m_tsst.group("part")
            meta.extras["file_ext"] = Path(path).suffix.lower().lstrip(".")
            return meta

        m = _R56_AV_STEM.match(stem)
        if m:
            family = m.group("family")
            meta.family_id = family
            meta.subject_id = f"{family}-dyad"
            meta.participant_type = "dyad"
            meta.task = _canonicalize_task(m.group("task"))
            if m.group("part") is not None:
                meta.extras["part"] = m.group("part")
            meta.extras["file_ext"] = Path(path).suffix.lower().lstrip(".")
            return meta

        # Fall back: recover family_id from the parent folder if filename was odd.
        for p in parts:
            if re.fullmatch(r"\d{4}", p):
                meta.family_id = p
                meta.subject_id = f"{p}-dyad"
                meta.participant_type = "dyad"
                break
        meta.extras["file_ext"] = Path(path).suffix.lower().lstrip(".")
        return meta

    return meta


def _classify_r01(path: str, modality: str, meta: RecordingMetadata) -> RecordingMetadata:
    # R01 is fNIRS-only at time of writing; keep the classifier lenient so that
    # if video/audio is added later under a recognizable pattern, nothing
    # outside this function has to change.
    parts = Path(path).parts
    meta.site = "PSU" if meta.study == "R01_PSU" else "WUSTL"
    meta.task = "DB-DOS"

    for p in parts:
        if p in ("T1", "T3", "T5"):
            meta.timepoint = p
            break

    for p in parts:
        m = re.fullmatch(r"(\d+)_([CP])", p)
        if m:
            family, ct = m.group(1), m.group(2)
            meta.family_id = family
            meta.subject_id = f"{family}_{ct}"
            meta.participant_type = "child" if ct == "C" else "adult"
            return meta

    return meta


register_study_classifier("CARE", _classify_care)
register_study_classifier("R56", _classify_r56)
register_study_classifier("R01_PSU", _classify_r01)
register_study_classifier("R01_WUSTL", _classify_r01)

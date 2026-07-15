"""
Minimal BIDS-ish dataset indexer.

The repo originally aimed to support multimodal data (fNIRS/video/audio). For the
diffusion fNIRS workflow we only need a reliable way to discover recordings on
disk. This module keeps the surface area small and avoids loading data eagerly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SessionIndex:
    fnirs_files: List[str]
    video_files: List[str]
    label_files: List[str]


class dataset:
    """
    Index a dataset folder laid out roughly like BIDS.

    Expected structure (flexible):
      bids_dir/sub-*/ses-*/**/*.{snirf,fif} or NIRX folder containing probeInfo.mat
    """

    def __init__(self, bids_dir: str):
        self.bids_dir = str(bids_dir)
        self.tracker: Dict[str, Dict[str, SessionIndex]] = {}
        self.subjects: List[str] = []
        self.sessions: List[str] = []

        # Modalities can be toggled by callers if needed.
        self.modalities = {"fnirs": True, "video": False, "audio": False}

    def orient(self, bids_dir: Optional[str] = None, training: bool = False) -> Dict[str, Dict[str, SessionIndex]]:
        if bids_dir:
            self.bids_dir = str(bids_dir)

        root = Path(self.bids_dir)
        if not root.exists():
            raise FileNotFoundError(f"BIDS dir not found: {self.bids_dir}")

        for sub_dir in sorted(root.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            sub = sub_dir.name
            self.tracker.setdefault(sub, {})
            if sub not in self.subjects:
                self.subjects.append(sub)

            for ses_dir in sorted(sub_dir.glob("ses-*")):
                if not ses_dir.is_dir():
                    continue
                ses = ses_dir.name
                if ses not in self.sessions:
                    self.sessions.append(ses)

                fnirs_files = self.find_fnirs_files(ses_dir) if self.modalities["fnirs"] else []
                video_files = self.find_video_files(ses_dir) if self.modalities["video"] else []
                label_files = self.find_label_files(ses_dir) if training else []

                self.tracker[sub][ses] = SessionIndex(
                    fnirs_files=fnirs_files,
                    video_files=video_files,
                    label_files=label_files,
                )

        return self.tracker

    def find_fnirs_files(self, session_folder: Path, extension: str = "all") -> List[str]:
        fnirs_paths: List[str] = []

        # NIRx directory is detected by presence of probeInfo.mat (common in exports).
        if extension in {"all", "nirx"}:
            for probe_info in session_folder.rglob("probeInfo.mat"):
                fnirs_paths.append(str(probe_info.parent))

        if extension in {"all", ".snirf"}:
            fnirs_paths.extend(str(p) for p in session_folder.rglob("*.snirf"))

        if extension in {"all", ".fif"}:
            fnirs_paths.extend(str(p) for p in session_folder.rglob("*.fif"))

        return sorted(set(fnirs_paths))

    def find_video_files(self, session_folder: Path, extension: str = ".mp4") -> List[str]:
        return sorted(str(p) for p in session_folder.rglob(f"*{extension}"))

    def find_label_files(self, session_folder: Path, extension: str = ".txt") -> List[str]:
        # Placeholder for future labels; keeps API stable.
        return sorted(str(p) for p in session_folder.rglob(f"*{extension}"))

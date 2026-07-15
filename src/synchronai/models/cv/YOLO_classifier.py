"""
Backwards-compatibility shim.

All video classifier code has moved to video_classifier.py.
This file re-exports everything for code that imports from YOLO_classifier.
"""

from synchronai.models.cv.video_classifier import *  # noqa: F401,F403
from synchronai.models.cv.video_classifier import (  # noqa: F401 — explicit re-exports
    VideoClassifierConfig,
    VideoClassifier,
    YOLOFeatureExtractor,
    TemporalAttention,
    TemporalLSTM,
    build_video_classifier,
    load_video_classifier,
)

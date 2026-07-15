"""Audio inference modules."""

from synchronai.inference.audio.predict import (
    AudioPrediction,
    AudioPredictionResult,
    export_predictions_csv,
    export_predictions_json,
    predict_audio,
    predict_audio_batch,
)

__all__ = [
    "AudioPrediction",
    "AudioPredictionResult",
    "export_predictions_csv",
    "export_predictions_json",
    "predict_audio",
    "predict_audio_batch",
]

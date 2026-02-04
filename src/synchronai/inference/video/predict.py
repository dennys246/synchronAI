"""
Video classification inference module.

Provides:
- Per-second synchrony predictions from video
- Batch inference for multiple videos
- Confidence scores and thresholding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from synchronai.data.video.processing import (
    VideoReaderPool,
    load_video_info,
    read_window_frames,
)
from synchronai.models.cv.YOLO_classifier import (
    VideoClassifier,
    VideoClassifierConfig,
    load_video_classifier,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result for a single video window."""

    second: int
    probability: float
    prediction: int
    confidence: float


@dataclass
class VideoPredictionResult:
    """Complete prediction result for a video."""

    video_path: str
    predictions: list[PredictionResult]
    overall_probability: float
    overall_prediction: int
    total_seconds: int
    synchrony_seconds: int
    synchrony_ratio: float


def predict_video(
    video_path: Union[str, Path],
    weights_path: Union[str, Path],
    threshold: float = 0.5,
    device: Optional[str] = None,
    config: Optional[VideoClassifierConfig] = None,
) -> VideoPredictionResult:
    """Run inference on a video file.

    Args:
        video_path: Path to video file
        weights_path: Path to model checkpoint
        threshold: Classification threshold
        device: Device to run on (auto-detected if None)
        config: Optional model config (loaded from checkpoint if None)

    Returns:
        VideoPredictionResult with per-second predictions
    """
    video_path = Path(video_path)
    weights_path = Path(weights_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, config = load_video_classifier(str(weights_path), config, device)
    model.eval()

    # Get video info
    video_info = load_video_info(str(video_path))
    total_seconds = int(video_info.duration)

    logger.info(
        f"Running inference on {video_path.name}: "
        f"{total_seconds}s @ {video_info.fps:.1f} FPS"
    )

    # Create reader pool
    reader_pool = VideoReaderPool(max_readers=1)
    reader = reader_pool.get_reader(str(video_path))

    predictions = []

    try:
        with torch.no_grad():
            for second in range(total_seconds):
                # Read frames for this second
                frames = read_window_frames(
                    video_path=str(video_path),
                    second=second,
                    sample_fps=config.sample_fps,
                    window_seconds=config.window_seconds,
                    target_size=config.frame_height,
                    reader=reader,
                )

                # Convert to tensor and add batch dimension
                frames_tensor = torch.from_numpy(frames).unsqueeze(0).to(device)

                # Run model
                logits = model(frames_tensor)
                prob = torch.sigmoid(logits).item()

                pred = 1 if prob >= threshold else 0
                confidence = prob if pred == 1 else 1 - prob

                predictions.append(
                    PredictionResult(
                        second=second,
                        probability=prob,
                        prediction=pred,
                        confidence=confidence,
                    )
                )

    finally:
        reader_pool.close_all()

    # Compute overall metrics
    probs = [p.probability for p in predictions]
    preds = [p.prediction for p in predictions]

    overall_prob = float(np.mean(probs))
    synchrony_seconds = sum(preds)
    synchrony_ratio = synchrony_seconds / len(predictions) if predictions else 0.0

    result = VideoPredictionResult(
        video_path=str(video_path),
        predictions=predictions,
        overall_probability=overall_prob,
        overall_prediction=1 if overall_prob >= threshold else 0,
        total_seconds=len(predictions),
        synchrony_seconds=synchrony_seconds,
        synchrony_ratio=synchrony_ratio,
    )

    logger.info(
        f"Inference complete: {synchrony_seconds}/{len(predictions)} seconds "
        f"classified as synchrony ({synchrony_ratio:.1%})"
    )

    return result


def predict_video_batch(
    video_paths: list[Union[str, Path]],
    weights_path: Union[str, Path],
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> list[VideoPredictionResult]:
    """Run inference on multiple videos.

    Args:
        video_paths: List of video file paths
        weights_path: Path to model checkpoint
        threshold: Classification threshold
        device: Device to run on

    Returns:
        List of VideoPredictionResult for each video
    """
    results = []
    for video_path in video_paths:
        try:
            result = predict_video(video_path, weights_path, threshold, device)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")

    return results


def export_predictions_csv(
    result: VideoPredictionResult,
    output_path: Union[str, Path],
) -> Path:
    """Export predictions to CSV file.

    Args:
        result: Prediction result
        output_path: Output CSV path

    Returns:
        Path to saved CSV
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["second", "probability", "prediction", "confidence"])
        for pred in result.predictions:
            writer.writerow(
                [pred.second, pred.probability, pred.prediction, pred.confidence]
            )

    logger.info(f"Exported predictions to {output_path}")
    return output_path

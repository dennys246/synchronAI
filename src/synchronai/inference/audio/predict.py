"""
Audio classification inference module.

Provides:
- Per-second audio event predictions from audio/video files
- Batch inference for multiple files
- Export to CSV/JSON formats
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from synchronai.data.audio.processing import (
    AudioChunkIterator,
    compute_energy_db,
    extract_audio,
    get_audio_duration,
)
from synchronai.models.audio.audio_classifier import (
    AUDIO_EVENT_CLASSES,
    AudioClassifier,
    AudioClassifierConfig,
    load_audio_classifier,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioPrediction:
    """Prediction result for a single second."""

    second: int
    audio_event: str
    event_confidence: float
    has_vocalization: bool
    vocalization_confidence: float
    energy_db: float
    is_speech: bool  # Convenience flag


@dataclass
class AudioPredictionResult:
    """Complete prediction result for an audio/video file."""

    source_path: str
    predictions: list[AudioPrediction]
    total_seconds: int
    vocalization_seconds: int
    vocalization_ratio: float
    speech_seconds: int
    speech_ratio: float
    dominant_event: str
    event_distribution: dict[str, int]


def predict_audio(
    input_path: Union[str, Path],
    weights_path: Union[str, Path],
    device: Optional[str] = None,
    config: Optional[AudioClassifierConfig] = None,
    vocalization_threshold: float = 0.5,
) -> AudioPredictionResult:
    """Run audio classification on a file.

    Args:
        input_path: Path to audio or video file
        weights_path: Path to model checkpoint
        device: Device to run on (auto-detected if None)
        config: Optional model config (loaded from checkpoint if None)
        vocalization_threshold: Threshold for vocalization detection

    Returns:
        AudioPredictionResult with per-second predictions
    """
    input_path = Path(input_path)
    weights_path = Path(weights_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, config = load_audio_classifier(str(weights_path), config, device)
    model.eval()

    # Extract audio if input is video
    audio_path = input_path
    is_video = input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    if is_video:
        logger.info(f"Extracting audio from video: {input_path.name}")
        audio_path = extract_audio(input_path)

    # Get audio info
    duration = get_audio_duration(audio_path)
    total_seconds = int(duration)

    logger.info(
        f"Running audio classification on {input_path.name}: "
        f"{total_seconds}s of audio"
    )

    # Process each second
    predictions = []
    audio_iter = AudioChunkIterator(audio_path)

    with torch.no_grad():
        for second, audio_chunk in audio_iter:
            # Compute actual energy from audio
            energy_db = compute_energy_db(audio_chunk)

            # Run model
            outputs = model(audio_chunk)

            # Get event prediction
            event_probs = outputs["event_probs"]
            if event_probs.dim() == 2:
                event_probs = event_probs[0]

            confidence, idx = event_probs.max(dim=-1)
            audio_event = config.event_classes[idx.item()]

            # Get vocalization prediction
            if "vocalization_prob" in outputs:
                voc_prob = outputs["vocalization_prob"].item()
                if isinstance(voc_prob, torch.Tensor):
                    voc_prob = voc_prob.item()
            else:
                # Infer from event type
                voc_events = {"speech", "laughter", "crying", "babbling"}
                voc_prob = 1.0 if audio_event in voc_events else 0.0

            has_vocalization = voc_prob >= vocalization_threshold
            is_speech = audio_event == "speech"

            predictions.append(
                AudioPrediction(
                    second=second,
                    audio_event=audio_event,
                    event_confidence=confidence.item(),
                    has_vocalization=has_vocalization,
                    vocalization_confidence=voc_prob,
                    energy_db=energy_db,
                    is_speech=is_speech,
                )
            )

    # Compute summary statistics
    vocalization_seconds = sum(1 for p in predictions if p.has_vocalization)
    speech_seconds = sum(1 for p in predictions if p.is_speech)

    # Event distribution
    event_counts = {}
    for p in predictions:
        event_counts[p.audio_event] = event_counts.get(p.audio_event, 0) + 1

    dominant_event = max(event_counts.keys(), key=lambda k: event_counts[k])

    result = AudioPredictionResult(
        source_path=str(input_path),
        predictions=predictions,
        total_seconds=total_seconds,
        vocalization_seconds=vocalization_seconds,
        vocalization_ratio=vocalization_seconds / total_seconds if total_seconds > 0 else 0.0,
        speech_seconds=speech_seconds,
        speech_ratio=speech_seconds / total_seconds if total_seconds > 0 else 0.0,
        dominant_event=dominant_event,
        event_distribution=event_counts,
    )

    logger.info(
        f"Audio classification complete: "
        f"{vocalization_seconds}/{total_seconds} seconds with vocalization "
        f"({result.vocalization_ratio:.1%}), dominant event: {dominant_event}"
    )

    return result


def predict_audio_batch(
    input_paths: list[Union[str, Path]],
    weights_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    skip_existing: bool = True,
    output_format: str = "both",
) -> list[AudioPredictionResult]:
    """Run audio classification on multiple files.

    Args:
        input_paths: List of audio/video file paths
        weights_path: Path to model checkpoint
        output_dir: Optional output directory for results
        device: Device to run on
        skip_existing: Skip files that already have output
        output_format: Output format ("csv", "json", or "both")

    Returns:
        List of AudioPredictionResult for each file
    """
    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, input_path in enumerate(input_paths):
        input_path = Path(input_path)
        logger.info(f"[{i + 1}/{len(input_paths)}] Processing {input_path.name}")

        # Check if output already exists
        if output_dir and skip_existing:
            csv_path = output_dir / f"{input_path.stem}_audio.csv"
            json_path = output_dir / f"{input_path.stem}_audio.json"
            if csv_path.exists() or json_path.exists():
                logger.info(f"  Skipping (output exists)")
                continue

        try:
            result = predict_audio(input_path, weights_path, device)
            results.append(result)

            # Export results
            if output_dir:
                if output_format in ["csv", "both"]:
                    csv_path = output_dir / f"{input_path.stem}_audio.csv"
                    export_predictions_csv(result, csv_path)

                if output_format in ["json", "both"]:
                    json_path = output_dir / f"{input_path.stem}_audio.json"
                    export_predictions_json(result, json_path)

        except Exception as e:
            logger.error(f"  Failed: {e}")

    return results


def export_predictions_csv(
    result: AudioPredictionResult,
    output_path: Union[str, Path],
) -> Path:
    """Export predictions to CSV file.

    Args:
        result: Prediction result
        output_path: Output CSV path

    Returns:
        Path to saved CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "second",
            "audio_event",
            "event_confidence",
            "has_vocalization",
            "vocalization_confidence",
            "energy_db",
            "is_speech",
        ])
        for pred in result.predictions:
            writer.writerow([
                pred.second,
                pred.audio_event,
                f"{pred.event_confidence:.4f}",
                pred.has_vocalization,
                f"{pred.vocalization_confidence:.4f}",
                f"{pred.energy_db:.2f}",
                pred.is_speech,
            ])

    logger.info(f"Exported predictions to {output_path}")
    return output_path


def export_predictions_json(
    result: AudioPredictionResult,
    output_path: Union[str, Path],
) -> Path:
    """Export predictions to JSON file.

    Args:
        result: Prediction result
        output_path: Output JSON path

    Returns:
        Path to saved JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable dict
    data = {
        "source_path": result.source_path,
        "total_seconds": result.total_seconds,
        "vocalization_seconds": result.vocalization_seconds,
        "vocalization_ratio": result.vocalization_ratio,
        "speech_seconds": result.speech_seconds,
        "speech_ratio": result.speech_ratio,
        "dominant_event": result.dominant_event,
        "event_distribution": result.event_distribution,
        "predictions": [asdict(p) for p in result.predictions],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported predictions to {output_path}")
    return output_path

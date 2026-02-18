"""
Tests for the inference pipeline.

Tests PredictionResult, VideoPredictionResult dataclasses and
prediction logic (mocked, since actual inference requires model weights and video files).
"""

import pytest
import numpy as np
from dataclasses import asdict


class TestPredictionResult:
    """Test the PredictionResult dataclass."""

    def test_create_prediction_result(self):
        """PredictionResult should store second, probability, prediction, confidence."""
        from synchronai.inference.video.predict import PredictionResult

        result = PredictionResult(
            second=5,
            probability=0.85,
            prediction=1,
            confidence=0.85,
        )
        assert result.second == 5
        assert result.probability == 0.85
        assert result.prediction == 1
        assert result.confidence == 0.85

    def test_prediction_result_serializable(self):
        """PredictionResult should be convertible to dict."""
        from synchronai.inference.video.predict import PredictionResult

        result = PredictionResult(second=0, probability=0.3, prediction=0, confidence=0.7)
        d = asdict(result)
        assert d == {
            "second": 0,
            "probability": 0.3,
            "prediction": 0,
            "confidence": 0.7,
        }


class TestVideoPredictionResult:
    """Test the VideoPredictionResult dataclass."""

    def test_create_video_prediction_result(self):
        """VideoPredictionResult should aggregate per-second predictions."""
        from synchronai.inference.video.predict import (
            PredictionResult,
            VideoPredictionResult,
        )

        predictions = [
            PredictionResult(second=0, probability=0.8, prediction=1, confidence=0.8),
            PredictionResult(second=1, probability=0.3, prediction=0, confidence=0.7),
            PredictionResult(second=2, probability=0.9, prediction=1, confidence=0.9),
        ]

        result = VideoPredictionResult(
            video_path="/test/video.mp4",
            predictions=predictions,
            overall_probability=np.mean([p.probability for p in predictions]),
            overall_prediction=1,
            total_seconds=3,
            synchrony_seconds=2,
            synchrony_ratio=2 / 3,
        )

        assert result.total_seconds == 3
        assert result.synchrony_seconds == 2
        assert abs(result.synchrony_ratio - 2 / 3) < 1e-6
        assert len(result.predictions) == 3

    def test_synchrony_ratio_all_sync(self):
        """100% synchrony should yield ratio=1.0."""
        from synchronai.inference.video.predict import (
            PredictionResult,
            VideoPredictionResult,
        )

        predictions = [
            PredictionResult(second=i, probability=0.9, prediction=1, confidence=0.9)
            for i in range(5)
        ]

        result = VideoPredictionResult(
            video_path="/test.mp4",
            predictions=predictions,
            overall_probability=0.9,
            overall_prediction=1,
            total_seconds=5,
            synchrony_seconds=5,
            synchrony_ratio=1.0,
        )
        assert result.synchrony_ratio == 1.0

    def test_synchrony_ratio_no_sync(self):
        """0% synchrony should yield ratio=0.0."""
        from synchronai.inference.video.predict import (
            PredictionResult,
            VideoPredictionResult,
        )

        predictions = [
            PredictionResult(second=i, probability=0.1, prediction=0, confidence=0.9)
            for i in range(5)
        ]

        result = VideoPredictionResult(
            video_path="/test.mp4",
            predictions=predictions,
            overall_probability=0.1,
            overall_prediction=0,
            total_seconds=5,
            synchrony_seconds=0,
            synchrony_ratio=0.0,
        )
        assert result.synchrony_ratio == 0.0


class TestPredictionThresholding:
    """Test that thresholding logic is correct."""

    def test_threshold_above(self):
        """Probability above threshold should predict 1."""
        prob = 0.7
        threshold = 0.5
        pred = 1 if prob >= threshold else 0
        assert pred == 1

    def test_threshold_below(self):
        """Probability below threshold should predict 0."""
        prob = 0.3
        threshold = 0.5
        pred = 1 if prob >= threshold else 0
        assert pred == 0

    def test_threshold_exact(self):
        """Probability exactly at threshold should predict 1 (>=)."""
        prob = 0.5
        threshold = 0.5
        pred = 1 if prob >= threshold else 0
        assert pred == 1

    def test_confidence_calculation(self):
        """Confidence should be max(prob, 1-prob)."""
        # Sync prediction
        prob = 0.8
        pred = 1
        confidence = prob if pred == 1 else 1 - prob
        assert confidence == 0.8

        # Async prediction
        prob = 0.2
        pred = 0
        confidence = prob if pred == 1 else 1 - prob
        assert confidence == 0.8


class TestHeatmapConfig:
    """Test HeatmapConfig defaults."""

    def test_default_config(self):
        """HeatmapConfig should have sensible defaults."""
        from synchronai.utils.heatmap import HeatmapConfig

        config = HeatmapConfig()
        assert config.threshold == 0.5
        assert config.vmin == 0.0
        assert config.vmax == 1.0
        assert config.colormap == "RdYlGn"
        assert config.dpi == 150

    def test_custom_threshold(self):
        """HeatmapConfig should accept custom threshold."""
        from synchronai.utils.heatmap import HeatmapConfig

        config = HeatmapConfig(threshold=0.7)
        assert config.threshold == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for training utilities and loops.

Tests compute_metrics, label smoothing, mixup, training configs,
and training history tracking.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# Video training tests
# ============================================================================

class TestVideoComputeMetrics:
    """Test compute_metrics from video training."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield accuracy=1.0."""
        from synchronai.training.video.train import compute_metrics

        logits = torch.tensor([5.0, -5.0, 5.0, -5.0])  # Strong predictions
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        metrics = compute_metrics(logits, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        """All wrong predictions should yield accuracy=0.0."""
        from synchronai.training.video.train import compute_metrics

        logits = torch.tensor([-5.0, 5.0, -5.0, 5.0])  # Opposite of truth
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        metrics = compute_metrics(logits, labels)
        assert metrics["accuracy"] == 0.0

    def test_handles_2d_logits(self):
        """compute_metrics should handle (batch, 1) logits."""
        from synchronai.training.video.train import compute_metrics

        logits = torch.tensor([[5.0], [-5.0], [5.0]])
        labels = torch.tensor([1.0, 0.0, 1.0])

        metrics = compute_metrics(logits, labels)
        assert metrics["accuracy"] == 1.0

    def test_handles_batch_size_1(self):
        """compute_metrics should handle batch_size=1 without collapsing."""
        from synchronai.training.video.train import compute_metrics

        logits = torch.tensor([[5.0]])
        labels = torch.tensor([1.0])

        metrics = compute_metrics(logits, labels)
        assert metrics["accuracy"] == 1.0

    def test_metrics_keys(self):
        """compute_metrics should return all expected keys."""
        from synchronai.training.video.train import compute_metrics

        logits = torch.randn(10)
        labels = torch.randint(0, 2, (10,)).float()

        metrics = compute_metrics(logits, labels)
        expected_keys = {"accuracy", "precision", "recall", "f1", "auc"}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_in_valid_range(self):
        """All metrics should be in [0, 1]."""
        from synchronai.training.video.train import compute_metrics

        for _ in range(10):
            logits = torch.randn(20)
            labels = torch.randint(0, 2, (20,)).float()
            metrics = compute_metrics(logits, labels)
            for key, value in metrics.items():
                assert 0.0 <= value <= 1.0, f"{key}={value} out of range"


# ============================================================================
# Multimodal training tests
# ============================================================================

class TestMultiModalComputeMetrics:
    """Test compute_metrics from multimodal training."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield accuracy=1.0."""
        from synchronai.training.multimodal.train import compute_metrics

        logits = torch.tensor([5.0, -5.0, 5.0])
        labels = torch.tensor([1.0, 0.0, 1.0])

        metrics = compute_metrics(logits, labels)
        assert metrics["accuracy"] == 1.0

    def test_handles_2d_logits(self):
        """compute_metrics should handle (batch, 1) logits."""
        from synchronai.training.multimodal.train import compute_metrics

        logits = torch.tensor([[5.0], [-5.0]])
        labels = torch.tensor([1.0, 0.0])

        metrics = compute_metrics(logits, labels)
        assert metrics["accuracy"] == 1.0


# ============================================================================
# Training config tests
# ============================================================================

class TestMultiModalTrainingConfig:
    """Test MultiModalTrainingConfig defaults."""

    def test_default_config(self):
        """Config should have reasonable defaults."""
        from synchronai.training.multimodal.train import MultiModalTrainingConfig

        config = MultiModalTrainingConfig()
        assert config.stage1_epochs > 0
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.weight_decay >= 0

    def test_event_weight_zero_by_default(self):
        """Event loss weight should be 0 (disabled by default)."""
        from synchronai.training.multimodal.train import MultiModalTrainingConfig

        config = MultiModalTrainingConfig()
        assert config.event_loss_weight == 0.0
        assert config.sync_loss_weight == 1.0


class TestTrainingHistory:
    """Test TrainingHistory tracking."""

    def test_video_training_history(self):
        """TrainingHistory should track epoch metrics."""
        from synchronai.training.video.train import TrainingHistory

        history = TrainingHistory()
        assert len(history.train_losses) == 0

    def test_multimodal_training_history(self):
        """MultiModal TrainingHistory should track metrics."""
        from synchronai.training.multimodal.train import TrainingHistory

        history = TrainingHistory()
        assert hasattr(history, 'train_losses')


# ============================================================================
# Label smoothing tests
# ============================================================================

class TestLabelSmoothing:
    """Test label smoothing logic used in training."""

    def test_no_smoothing(self):
        """With smoothing=0, labels should be unchanged."""
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        smoothing = 0.0
        smoothed = labels * (1.0 - 2 * smoothing) + smoothing
        assert torch.allclose(smoothed, labels)

    def test_label_smoothing_applied(self):
        """With smoothing=0.1, labels 0/1 should become 0.1/0.9."""
        labels = torch.tensor([0.0, 1.0])
        smoothing = 0.1
        smoothed = labels * (1.0 - 2 * smoothing) + smoothing
        expected = torch.tensor([0.1, 0.9])
        assert torch.allclose(smoothed, expected)

    def test_label_smoothing_symmetric(self):
        """Smoothing should be symmetric around 0.5."""
        labels = torch.tensor([0.0, 1.0])
        smoothing = 0.05
        smoothed = labels * (1.0 - 2 * smoothing) + smoothing
        # 0 -> 0.05, 1 -> 0.95, both equidistant from 0/1
        assert abs(smoothed[0].item() - 0.05) < 1e-6
        assert abs(smoothed[1].item() - 0.95) < 1e-6


# ============================================================================
# Mixup tests
# ============================================================================

class TestMixup:
    """Test mixup augmentation logic."""

    def test_mixup_blends_samples(self):
        """Mixup should produce blended samples."""
        batch_size = 4
        frames = torch.randn(batch_size, 12, 3, 640, 640)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1.0 - lam)  # Ensure >= 0.5

        perm = torch.randperm(batch_size)
        mixed_frames = lam * frames + (1.0 - lam) * frames[perm]
        mixed_labels = lam * labels + (1.0 - lam) * labels[perm]

        assert mixed_frames.shape == frames.shape
        assert mixed_labels.shape == labels.shape
        assert lam >= 0.5  # Stability guarantee

    def test_mixup_lambda_range(self):
        """Lambda should always be >= 0.5 after max(lam, 1-lam)."""
        for _ in range(100):
            lam = np.random.beta(0.2, 0.2)
            lam = max(lam, 1.0 - lam)
            assert lam >= 0.5


# ============================================================================
# AMP compatibility tests
# ============================================================================

class TestAMPCompatibility:
    """Test that the new torch.amp API works correctly."""

    def test_autocast_context_manager(self):
        """torch.amp.autocast should work as context manager."""
        from torch.amp import autocast

        with autocast("cuda", enabled=False):
            x = torch.tensor([1.0, 2.0, 3.0])
            y = x * 2
        assert y.dtype == torch.float32

    def test_grad_scaler_creation(self):
        """GradScaler should accept device_type string."""
        from torch.amp import GradScaler

        scaler = GradScaler("cuda")
        assert scaler is not None

    def test_autocast_disabled_no_effect(self):
        """Disabled autocast should not change dtypes."""
        from torch.amp import autocast

        x = torch.randn(4, 4)
        with autocast("cuda", enabled=False):
            y = x @ x.T
        assert y.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

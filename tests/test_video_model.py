"""
Unit tests for the video classifier model (DINOv2 / YOLO backbone + temporal aggregation).

Tests cover:
- VideoClassifierConfig: defaults, n_frames property, custom values
- TemporalAttention: forward shape, attention weight normalization
- TemporalLSTM: forward shape, output dimension
- YOLOFeatureExtractor: construction with mocked YOLO, feature extraction
- DINOv2 backend: construction with mocked DINOv2, feature extraction
- VideoClassifier: full forward pass, return_features mode,
  freeze/unfreeze backbone, get_parameter_groups filtering,
  all temporal aggregation modes (mean, max, attention, lstm)

The YOLO/ultralytics dependency is mocked throughout so these tests
run without the ultralytics package installed. DINOv2 tests mock
the transformers library similarly.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from synchronai.models.cv.video_classifier import (
    TemporalAttention,
    TemporalLSTM,
    VideoClassifier,
    VideoClassifierConfig,
    YOLOFeatureExtractor,
    build_video_classifier,
)


# ---------------------------------------------------------------------------
# Helpers: mock YOLO backbone
# ---------------------------------------------------------------------------

MOCK_FEATURE_DIM = 512


class FakeBackboneLayer(nn.Module):
    """A single fake backbone layer that preserves spatial dims but sets channels."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept any number of input channels by just projecting via our conv.
        # If the input has != 3 channels we need to adapt, but in practice we
        # stack 10 identical layers -- so only the first one sees 3 channels.
        if x.shape[1] != self.conv.in_channels:
            # shortcut: treat later layers as identity-like (resize channels)
            return x[:, :self.conv.out_channels]
        return self.conv(x)


def _build_fake_backbone(feature_dim: int = MOCK_FEATURE_DIM) -> nn.ModuleList:
    """Return a 10-layer nn.ModuleList that mimics YOLO backbone slicing.

    Each layer is a simple conv that maps (B, C_in, H, W) -> (B, feature_dim, H, W).
    The real VideoClassifier applies adaptive_avg_pool2d afterwards.
    """
    layers = nn.ModuleList()
    for i in range(10):
        layer = nn.Conv2d(
            in_channels=3 if i == 0 else feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
            bias=False,
        )
        layers.append(layer)
    return layers


def _make_mock_yolo(feature_dim: int = MOCK_FEATURE_DIM) -> MagicMock:
    """Create a MagicMock that looks like an ultralytics.YOLO instance.

    `mock.model.model[:10]` returns a usable nn.ModuleList so that
    YOLOFeatureExtractor can iterate and call `layer(x)`.
    """
    backbone = _build_fake_backbone(feature_dim)

    inner_model = MagicMock()
    inner_model.model = backbone  # model.model.model[:10] -> backbone[:10]

    yolo_mock = MagicMock()
    yolo_mock.model = inner_model
    return yolo_mock


@pytest.fixture()
def patch_yolo():
    """Fixture that patches ultralytics.YOLO so imports don't fail.

    Yields the mock YOLO class so tests can inspect calls if needed.
    """
    mock_yolo_cls = MagicMock(side_effect=lambda *a, **kw: _make_mock_yolo())
    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        yield mock_yolo_cls


@pytest.fixture()
def default_config() -> VideoClassifierConfig:
    """Return a default VideoClassifierConfig."""
    return VideoClassifierConfig()


# ===========================================================================
# VideoClassifierConfig
# ===========================================================================

class TestVideoClassifierConfig:
    """Tests for VideoClassifierConfig dataclass."""

    def test_defaults(self):
        """Default values should match the documented architecture."""
        cfg = VideoClassifierConfig()
        assert cfg.window_seconds == 2.0
        assert cfg.sample_fps == 12.0
        assert cfg.frame_height == 224
        assert cfg.frame_width == 224
        assert cfg.backbone == "dinov2-base"
        assert cfg.backbone_task == "detect"
        assert cfg.backbone_weights is None
        assert cfg.freeze_backbone is True
        assert cfg.temporal_aggregation == "lstm"
        assert cfg.hidden_dim == 256
        assert cfg.dropout == 0.3
        assert cfg.output_dim == 1
        assert cfg.weights_path == "video_classifier.pt"
        assert cfg.gradient_checkpointing is False

    def test_n_frames_default(self):
        """n_frames = sample_fps * window_seconds = 12 * 2 = 24."""
        cfg = VideoClassifierConfig()
        assert cfg.n_frames == 24

    def test_n_frames_custom(self):
        """n_frames with non-default fps and window."""
        cfg = VideoClassifierConfig(window_seconds=5.0, sample_fps=30.0)
        assert cfg.n_frames == 150

    def test_n_frames_fractional_truncation(self):
        """n_frames truncates fractional frames via int()."""
        cfg = VideoClassifierConfig(window_seconds=1.0, sample_fps=7.5)
        # 7.5 * 1.0 = 7.5 -> int(7.5) = 7
        assert cfg.n_frames == 7

    def test_replace_creates_new_instance(self):
        """dataclasses.replace should produce independent copies."""
        cfg1 = VideoClassifierConfig()
        cfg2 = replace(cfg1, hidden_dim=128)
        assert cfg1.hidden_dim == 256
        assert cfg2.hidden_dim == 128


# ===========================================================================
# TemporalAttention
# ===========================================================================

class TestTemporalAttention:
    """Tests for the TemporalAttention module."""

    def test_output_shape(self):
        """Output should be (batch, feature_dim) regardless of n_frames."""
        feature_dim = 64
        batch, n_frames = 4, 10
        module = TemporalAttention(feature_dim=feature_dim, hidden_dim=32)

        x = torch.randn(batch, n_frames, feature_dim)
        out = module(x)
        assert out.shape == (batch, feature_dim)

    def test_output_shape_single_frame(self):
        """With a single frame the attention should still work."""
        feature_dim = 32
        module = TemporalAttention(feature_dim=feature_dim)

        x = torch.randn(1, 1, feature_dim)
        out = module(x)
        assert out.shape == (1, feature_dim)

    def test_attention_weights_sum_to_one(self):
        """Softmax attention weights over frames must sum to 1."""
        feature_dim = 16
        module = TemporalAttention(feature_dim=feature_dim, hidden_dim=8)

        x = torch.randn(2, 5, feature_dim)
        # Manually compute attention weights the same way the module does
        attn_weights = module.attention(x)  # (2, 5, 1)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)

        sums = attn_weights.sum(dim=1)  # (2, 1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows(self):
        """Gradients should flow back through attention."""
        feature_dim = 16
        module = TemporalAttention(feature_dim=feature_dim)
        x = torch.randn(2, 4, feature_dim, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ===========================================================================
# TemporalLSTM
# ===========================================================================

class TestTemporalLSTM:
    """Tests for the TemporalLSTM module."""

    def test_output_shape(self):
        """Output should be (batch, hidden_dim)."""
        feature_dim = 64
        hidden_dim = 128
        module = TemporalLSTM(feature_dim=feature_dim, hidden_dim=hidden_dim)

        x = torch.randn(4, 10, feature_dim)
        out = module(x)
        assert out.shape == (4, hidden_dim)

    def test_output_dim_attribute(self):
        """TemporalLSTM.output_dim should match hidden_dim."""
        module = TemporalLSTM(feature_dim=32, hidden_dim=64)
        assert module.output_dim == 64

    def test_single_frame(self):
        """LSTM should handle a single-frame sequence."""
        feature_dim = 16
        hidden_dim = 32
        module = TemporalLSTM(feature_dim=feature_dim, hidden_dim=hidden_dim)

        x = torch.randn(1, 1, feature_dim)
        out = module(x)
        assert out.shape == (1, hidden_dim)

    def test_gradient_flows(self):
        """Gradients should flow back through LSTM."""
        feature_dim = 16
        hidden_dim = 32
        module = TemporalLSTM(feature_dim=feature_dim, hidden_dim=hidden_dim)
        x = torch.randn(2, 5, feature_dim, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ===========================================================================
# YOLOFeatureExtractor (mocked)
# ===========================================================================

class TestYOLOFeatureExtractor:
    """Tests for YOLOFeatureExtractor with mocked ultralytics."""

    def test_construction_default_weights(self, patch_yolo):
        """Should construct without error using default pretrained weights."""
        extractor = YOLOFeatureExtractor(
            backbone="yolo26s", task="detect", weights_path=None, freeze=True
        )
        assert extractor.backbone is not None

    def test_construction_custom_weights(self, patch_yolo):
        """Should construct when a custom weights_path is provided."""
        extractor = YOLOFeatureExtractor(
            backbone="yolo26s", task="detect",
            weights_path="/tmp/custom.pt", freeze=False,
        )
        assert extractor.backbone is not None

    def test_feature_dim_property(self, patch_yolo):
        """feature_dim should be computed lazily and match the mock."""
        extractor = YOLOFeatureExtractor(freeze=False)
        dim = extractor.feature_dim
        assert isinstance(dim, int)
        assert dim == MOCK_FEATURE_DIM

    def test_forward_shape(self, patch_yolo):
        """Forward should return (batch, feature_dim)."""
        extractor = YOLOFeatureExtractor(freeze=False)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = extractor(x)
        assert out.shape == (2, MOCK_FEATURE_DIM)

    def test_freeze_sets_requires_grad_false(self, patch_yolo):
        """When freeze=True, all backbone params should have requires_grad=False."""
        extractor = YOLOFeatureExtractor(freeze=True)
        for p in extractor.backbone.parameters():
            assert p.requires_grad is False

    def test_no_freeze_keeps_requires_grad_true(self, patch_yolo):
        """When freeze=False, backbone params should remain trainable."""
        extractor = YOLOFeatureExtractor(freeze=False)
        trainable = [p for p in extractor.backbone.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_pose_model_name(self, patch_yolo):
        """Pose task should form model name as '{backbone}-pose.pt'."""
        _ = YOLOFeatureExtractor(backbone="yolo26s", task="pose", freeze=False)
        # The mock YOLO class was called; verify the model name
        call_args = patch_yolo.call_args
        assert "yolo26s-pose.pt" in str(call_args)


# ===========================================================================
# VideoClassifier (full model, mocked YOLO)
# ===========================================================================

class TestVideoClassifier:
    """Tests for the full VideoClassifier model with mocked YOLO."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_yolo):
        """Auto-patch YOLO for every test in this class."""

    def _make_model(self, **config_overrides) -> VideoClassifier:
        """Helper to build a YOLO-backed VideoClassifier with optional config overrides."""
        defaults = {"backbone": "yolo26s", "frame_height": 64, "frame_width": 64}
        defaults.update(config_overrides)
        cfg = VideoClassifierConfig(**defaults)
        return VideoClassifier(cfg)

    # --- Forward pass -------------------------------------------------------

    def test_forward_shape_lstm(self):
        """Forward with LSTM temporal aggregation returns (batch, output_dim)."""
        model = self._make_model(temporal_aggregation="lstm")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    def test_forward_shape_attention(self):
        """Forward with attention temporal aggregation returns correct shape."""
        model = self._make_model(temporal_aggregation="attention")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    def test_forward_shape_mean(self):
        """Forward with mean temporal aggregation returns correct shape."""
        model = self._make_model(temporal_aggregation="mean")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    def test_forward_shape_max(self):
        """Forward with max temporal aggregation returns correct shape."""
        model = self._make_model(temporal_aggregation="max")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    def test_invalid_temporal_aggregation(self):
        """Unknown temporal aggregation should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown temporal aggregation"):
            self._make_model(temporal_aggregation="transformer")

    # --- return_features ----------------------------------------------------

    def test_return_features_keys(self):
        """return_features=True should return dict with expected keys."""
        model = self._make_model(temporal_aggregation="lstm")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            result = model(x, return_features=True)

        assert isinstance(result, dict)
        assert "logits" in result
        assert "temporal_features" in result
        assert "frame_features" in result

    def test_return_features_logits_shape(self):
        """Logits in the features dict should have shape (batch, output_dim)."""
        model = self._make_model(temporal_aggregation="lstm")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            result = model(x, return_features=True)

        assert result["logits"].shape == (batch, model.config.output_dim)

    def test_return_features_temporal_shape_lstm(self):
        """temporal_features with LSTM should be (batch, hidden_dim)."""
        model = self._make_model(temporal_aggregation="lstm", hidden_dim=128)
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            result = model(x, return_features=True)

        assert result["temporal_features"].shape == (batch, 128)

    def test_return_features_temporal_shape_attention(self):
        """temporal_features with attention should be (batch, feature_dim)."""
        model = self._make_model(temporal_aggregation="attention")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            result = model(x, return_features=True)

        # Attention preserves feature_dim (not hidden_dim)
        assert result["temporal_features"].shape == (batch, MOCK_FEATURE_DIM)

    def test_return_features_frame_features_shape(self):
        """frame_features should be (batch, n_frames, feature_dim)."""
        model = self._make_model(temporal_aggregation="lstm")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 64, 64)
        with torch.no_grad():
            result = model(x, return_features=True)

        assert result["frame_features"].shape == (batch, n_frames, MOCK_FEATURE_DIM)

    # --- Freeze / unfreeze --------------------------------------------------

    def test_freeze_backbone(self):
        """freeze_backbone() sets all backbone params to requires_grad=False."""
        model = self._make_model(freeze_backbone=False)
        model.freeze_backbone()

        for p in model.feature_extractor.backbone.parameters():
            assert p.requires_grad is False

    def test_unfreeze_backbone_all(self):
        """unfreeze_backbone('all') sets all backbone params trainable."""
        model = self._make_model(freeze_backbone=True)
        model.unfreeze_backbone(mode="all")

        for p in model.feature_extractor.backbone.parameters():
            assert p.requires_grad is True

    def test_unfreeze_backbone_last(self):
        """unfreeze_backbone('last') unfreezes only the last 3 layers."""
        model = self._make_model(freeze_backbone=True)
        model.unfreeze_backbone(mode="last")

        layers = list(model.feature_extractor.backbone.children())
        # Last 3 layers should be trainable
        for layer in layers[-3:]:
            for p in layer.parameters():
                assert p.requires_grad is True
        # Earlier layers should remain frozen
        for layer in layers[:-3]:
            for p in layer.parameters():
                assert p.requires_grad is False

    def test_freeze_then_unfreeze_roundtrip(self):
        """Freezing then unfreezing should restore trainability."""
        model = self._make_model(freeze_backbone=False)

        # All start trainable
        trainable_before = sum(
            1 for p in model.feature_extractor.backbone.parameters() if p.requires_grad
        )
        assert trainable_before > 0

        model.freeze_backbone()
        trainable_frozen = sum(
            1 for p in model.feature_extractor.backbone.parameters() if p.requires_grad
        )
        assert trainable_frozen == 0

        model.unfreeze_backbone(mode="all")
        trainable_after = sum(
            1 for p in model.feature_extractor.backbone.parameters() if p.requires_grad
        )
        assert trainable_after == trainable_before

    # --- get_parameter_groups -----------------------------------------------

    def test_get_parameter_groups_frozen_backbone(self):
        """When backbone is frozen, get_parameter_groups should NOT include backbone params."""
        model = self._make_model(freeze_backbone=True, temporal_aggregation="lstm")
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        # Frozen backbone means no backbone group; only head + temporal
        for group in groups:
            for p in group["params"]:
                # All returned params must require grad
                if isinstance(p, torch.Tensor):
                    assert p.requires_grad is True

        # There should be no group with backbone_lr when backbone is frozen
        lrs = [g["lr"] for g in groups]
        assert 1e-5 not in lrs

    def test_get_parameter_groups_unfrozen_backbone(self):
        """When backbone is unfrozen, get_parameter_groups includes backbone at backbone_lr."""
        model = self._make_model(freeze_backbone=False, temporal_aggregation="lstm")
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        lrs = [g["lr"] for g in groups]
        assert 1e-5 in lrs  # backbone group present
        assert 1e-3 in lrs  # head group present

    def test_get_parameter_groups_only_requires_grad(self):
        """Every param returned by get_parameter_groups must have requires_grad=True."""
        model = self._make_model(freeze_backbone=False, temporal_aggregation="lstm")
        # Partially freeze: freeze backbone, keep head trainable
        model.freeze_backbone()
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        for group in groups:
            for p in group["params"]:
                if isinstance(p, torch.Tensor):
                    assert p.requires_grad, (
                        "get_parameter_groups must not return frozen params"
                    )

    def test_get_parameter_groups_lstm_temporal_included(self):
        """With LSTM temporal, there should be a group for temporal params."""
        model = self._make_model(
            freeze_backbone=True, temporal_aggregation="lstm"
        )
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        # Should have head group + temporal group (no backbone since frozen)
        assert len(groups) == 2

    def test_get_parameter_groups_mean_no_temporal_group(self):
        """With mean aggregation (no temporal module), there is no temporal group."""
        model = self._make_model(
            freeze_backbone=True, temporal_aggregation="mean"
        )
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        # Only head group (no backbone since frozen, no temporal since mean)
        assert len(groups) == 1
        assert groups[0]["lr"] == 1e-3

    # --- Miscellaneous ------------------------------------------------------

    def test_temporal_is_none_for_mean(self):
        """Mean aggregation should set self.temporal = None."""
        model = self._make_model(temporal_aggregation="mean")
        assert model.temporal is None

    def test_temporal_is_none_for_max(self):
        """Max aggregation should set self.temporal = None."""
        model = self._make_model(temporal_aggregation="max")
        assert model.temporal is None

    def test_temporal_is_attention_module(self):
        """Attention aggregation should use TemporalAttention."""
        model = self._make_model(temporal_aggregation="attention")
        assert isinstance(model.temporal, TemporalAttention)

    def test_temporal_is_lstm_module(self):
        """LSTM aggregation should use TemporalLSTM."""
        model = self._make_model(temporal_aggregation="lstm")
        assert isinstance(model.temporal, TemporalLSTM)

    def test_head_structure(self):
        """Head should be an nn.Sequential with Linear, BN, ReLU, Dropout layers."""
        model = self._make_model(temporal_aggregation="lstm")
        head = model.head
        assert isinstance(head, nn.Sequential)
        # Check final layer outputs correct dim
        last_linear = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
        assert last_linear.out_features == model.config.output_dim

    def test_batch_size_one(self):
        """Model should handle batch_size=1 without squeeze issues."""
        model = self._make_model(temporal_aggregation="lstm")
        model.eval()  # BatchNorm1d requires batch>1 in train mode
        x = torch.randn(1, model.config.n_frames, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, model.config.output_dim)

    def test_custom_output_dim(self):
        """Model with output_dim > 1 (multi-class) should produce correct shape."""
        model = self._make_model(
            temporal_aggregation="lstm", output_dim=5,
        )
        batch = 2
        x = torch.randn(batch, model.config.n_frames, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, 5)


# ===========================================================================
# build_video_classifier helper
# ===========================================================================

class TestBuildVideoClassifier:
    """Tests for the build_video_classifier factory function."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_yolo):
        """Auto-patch YOLO."""

    def test_returns_video_classifier(self):
        """build_video_classifier should return a VideoClassifier instance."""
        cfg = VideoClassifierConfig(backbone="yolo26s", frame_height=64, frame_width=64)
        model = build_video_classifier(cfg)
        assert isinstance(model, VideoClassifier)
        assert model.config is cfg


# ===========================================================================
# DINOv2 backend tests
# ===========================================================================

DINOV2_FEATURE_DIM = 768  # dinov2-base hidden size


class FakeDINOv2Output:
    """Mimics the output of HuggingFace Dinov2Model."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class FakeDINOv2(nn.Module):
    """Lightweight stand-in for HuggingFace Dinov2Model."""

    def __init__(self, hidden_size: int = DINOV2_FEATURE_DIM):
        super().__init__()
        self.hidden_size = hidden_size
        # Minimal parameters so freeze/unfreeze have something to work with
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, pixel_values=None, **kwargs):
        B = pixel_values.shape[0]
        # 1 CLS token + 256 patch tokens = 257
        hidden = torch.randn(B, 257, self.hidden_size, device=pixel_values.device)
        return FakeDINOv2Output(hidden)


@pytest.fixture()
def patch_dinov2():
    """Fixture that patches transformers.Dinov2Model so imports don't fail.

    Yields the mock Dinov2Model class for inspection.
    """
    fake_model = FakeDINOv2()

    mock_dinov2_cls = MagicMock()
    mock_dinov2_cls.from_pretrained = MagicMock(return_value=fake_model)

    mock_transformers = MagicMock()
    mock_transformers.Dinov2Model = mock_dinov2_cls

    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        yield mock_dinov2_cls


class TestDINOv2VideoClassifier:
    """Tests for VideoClassifier with DINOv2 backend (mocked transformers)."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_dinov2):
        """Auto-patch DINOv2 for every test in this class."""

    def _make_model(self, **config_overrides) -> VideoClassifier:
        """Helper to build a DINOv2-backed VideoClassifier."""
        defaults = {"backbone": "dinov2-base", "frame_height": 224, "frame_width": 224}
        defaults.update(config_overrides)
        cfg = VideoClassifierConfig(**defaults)
        return VideoClassifier(cfg)

    # --- Forward pass -------------------------------------------------------

    def test_forward_shape_lstm(self):
        """Forward with DINOv2 + LSTM should return (batch, output_dim)."""
        model = self._make_model(temporal_aggregation="lstm")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    def test_forward_shape_attention(self):
        """Forward with DINOv2 + attention should return correct shape."""
        model = self._make_model(temporal_aggregation="attention")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    def test_forward_shape_mean(self):
        """Forward with DINOv2 + mean aggregation should return correct shape."""
        model = self._make_model(temporal_aggregation="mean")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    def test_forward_shape_max(self):
        """Forward with DINOv2 + max aggregation should return correct shape."""
        model = self._make_model(temporal_aggregation="max")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, model.config.output_dim)

    # --- Feature dim --------------------------------------------------------

    def test_feature_dim(self):
        """DINOv2-base should report feature_dim=768."""
        model = self._make_model()
        assert model.feature_extractor.feature_dim == DINOV2_FEATURE_DIM

    def test_feature_dim_small(self):
        """DINOv2-small should report feature_dim=384."""
        model = self._make_model(backbone="dinov2-small")
        assert model.feature_extractor.feature_dim == 384

    # --- return_features ----------------------------------------------------

    def test_return_features_keys(self):
        """return_features=True should return dict with expected keys."""
        model = self._make_model(temporal_aggregation="lstm")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 224, 224)
        with torch.no_grad():
            result = model(x, return_features=True)

        assert isinstance(result, dict)
        assert "logits" in result
        assert "temporal_features" in result
        assert "frame_features" in result

    def test_return_features_frame_features_shape(self):
        """frame_features should be (batch, n_frames, feature_dim)."""
        model = self._make_model(temporal_aggregation="lstm")
        batch, n_frames = 2, model.config.n_frames
        x = torch.randn(batch, n_frames, 3, 224, 224)
        with torch.no_grad():
            result = model(x, return_features=True)

        assert result["frame_features"].shape == (batch, n_frames, DINOV2_FEATURE_DIM)

    # --- Freeze / unfreeze --------------------------------------------------

    def test_freeze_backbone(self):
        """freeze_backbone() sets all DINOv2 params to requires_grad=False."""
        model = self._make_model(freeze_backbone=False)
        # Trigger lazy loading so dinov2 is not None
        model.feature_extractor._load_model()
        model.freeze_backbone()

        for p in model.feature_extractor.dinov2.parameters():
            assert p.requires_grad is False

    def test_unfreeze_backbone(self):
        """unfreeze_backbone() sets all DINOv2 params trainable."""
        model = self._make_model(freeze_backbone=True)
        # Trigger lazy loading so dinov2 is not None
        model.feature_extractor._load_model()
        model.unfreeze_backbone(mode="all")

        for p in model.feature_extractor.dinov2.parameters():
            assert p.requires_grad is True

    # --- get_parameter_groups -----------------------------------------------

    def test_get_parameter_groups_frozen_backbone(self):
        """When DINOv2 backbone is frozen, no backbone params in groups."""
        model = self._make_model(freeze_backbone=True, temporal_aggregation="lstm")
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        # All returned params must require grad
        for group in groups:
            for p in group["params"]:
                if isinstance(p, torch.Tensor):
                    assert p.requires_grad is True

        # No backbone LR group
        lrs = [g["lr"] for g in groups]
        assert 1e-5 not in lrs

    def test_get_parameter_groups_unfrozen_backbone(self):
        """When DINOv2 backbone is unfrozen, backbone params included at backbone_lr."""
        model = self._make_model(freeze_backbone=False, temporal_aggregation="lstm")
        # Trigger lazy loading so dinov2 params are available
        model.feature_extractor._load_model()
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        lrs = [g["lr"] for g in groups]
        assert 1e-5 in lrs  # backbone group present
        assert 1e-3 in lrs  # head group present

    # --- Batch size edge case -----------------------------------------------

    def test_batch_size_one(self):
        """Model should handle batch_size=1."""
        model = self._make_model(temporal_aggregation="lstm")
        model.eval()  # BatchNorm1d requires batch>1 in train mode
        x = torch.randn(1, model.config.n_frames, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, model.config.output_dim)

    def test_custom_output_dim(self):
        """Model with output_dim > 1 should produce correct shape."""
        model = self._make_model(temporal_aggregation="lstm", output_dim=5)
        batch = 2
        x = torch.randn(batch, model.config.n_frames, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, 5)


class TestBuildVideoClassifierDINOv2:
    """Tests for build_video_classifier with DINOv2 backend."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_dinov2):
        """Auto-patch DINOv2."""

    def test_returns_video_classifier(self):
        """build_video_classifier should return a VideoClassifier with DINOv2."""
        cfg = VideoClassifierConfig(backbone="dinov2-base")
        model = build_video_classifier(cfg)
        assert isinstance(model, VideoClassifier)
        assert model.config is cfg

    def test_unknown_backbone_raises(self):
        """Unknown backbone prefix should raise ValueError."""
        cfg = VideoClassifierConfig(backbone="resnet50")
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_video_classifier(cfg)

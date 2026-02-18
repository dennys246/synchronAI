"""
Comprehensive tests for the audio classifier model.

Tests cover:
- WhisperEncoderConfig: dataclass fields, encoder_dim property for all model sizes
- WhisperEncoderFeatures: lazy loading, extract_features pooling, encoder_dim property
- AudioClassifierConfig: defaults, properties, encoder_dim delegation
- AudioClassifier: forward output shapes, freeze/unfreeze, get_parameter_groups,
  predict_event, optional heads, build/load/save helpers

The Whisper model itself is mocked throughout to avoid requiring the
openai-whisper package or downloading model weights.
"""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_whisper_cache_dir(tmp_path):
    """Prevent tests from touching the real Whisper cache directory."""
    with patch(
        "synchronai.models.audio.whisper_encoder.get_whisper_cache_dir",
        return_value=tmp_path / "whisper_cache",
    ):
        yield


def _make_fake_encoder(encoder_dim: int = 1280, total_frames: int = 1500):
    """Create a fake encoder nn.Module that mimics Whisper encoder output.

    Returns a module whose forward produces (batch, total_frames, encoder_dim).
    """
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # A real parameter so that parameters() is non-empty
            self.proj = nn.Linear(encoder_dim, encoder_dim)
            self._dim = encoder_dim
            self._frames = total_frames

        def forward(self, mel):
            batch = mel.shape[0]
            return torch.randn(batch, self._frames, self._dim)

    return FakeEncoder()


def _build_classifier_with_fake_encoder(config=None, encoder_dim=1280, total_frames=1500):
    """Build an AudioClassifier whose WhisperEncoderFeatures uses a fake encoder.

    This avoids loading the real Whisper model while keeping the rest of the
    AudioClassifier logic intact.
    """
    from synchronai.models.audio.audio_classifier import AudioClassifier, AudioClassifierConfig

    if config is None:
        config = AudioClassifierConfig(whisper_model_size="large-v3")

    # Determine the encoder dim the config would expect
    from synchronai.models.audio.whisper_encoder import WHISPER_DIMS
    base_size = config.whisper_model_size.split("-")[0]
    expected_dim = WHISPER_DIMS.get(base_size, WHISPER_DIMS.get(config.whisper_model_size, 1280))

    # Patch _load_model so it does not call whisper.load_model
    with patch(
        "synchronai.models.audio.whisper_encoder.WhisperEncoderFeatures._load_model",
        autospec=True,
    ) as mock_load:
        def side_effect(self_arg):
            self_arg.encoder = _make_fake_encoder(expected_dim, total_frames)
            self_arg._is_loaded = True

        mock_load.side_effect = side_effect

        model = AudioClassifier(config)

        # Trigger the fake load so that the encoder is populated
        model.encoder._load_model()

    return model


# ============================================================================
# WhisperEncoderConfig tests
# ============================================================================

class TestWhisperEncoderConfig:
    """Tests for WhisperEncoderConfig dataclass and its encoder_dim property."""

    def test_default_values(self):
        from synchronai.models.audio.whisper_encoder import WhisperEncoderConfig

        cfg = WhisperEncoderConfig()
        assert cfg.model_size == "large-v3"
        assert cfg.device is None
        assert cfg.freeze is True
        assert cfg.chunk_duration == 1.0
        assert cfg.sample_rate == 16000

    @pytest.mark.parametrize(
        "model_size, expected_dim",
        [
            ("tiny", 384),
            ("base", 512),
            ("small", 768),
            ("medium", 1024),
            ("large", 1280),
            ("large-v2", 1280),
            ("large-v3", 1280),
        ],
    )
    def test_encoder_dim_for_all_model_sizes(self, model_size, expected_dim):
        from synchronai.models.audio.whisper_encoder import WhisperEncoderConfig

        cfg = WhisperEncoderConfig(model_size=model_size)
        assert cfg.encoder_dim == expected_dim

    def test_encoder_dim_unknown_size_defaults_to_1280(self):
        """Unknown model sizes should fall back to 1280."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderConfig

        cfg = WhisperEncoderConfig(model_size="turbo-9000")
        assert cfg.encoder_dim == 1280

    def test_custom_values(self):
        from synchronai.models.audio.whisper_encoder import WhisperEncoderConfig

        cfg = WhisperEncoderConfig(
            model_size="small",
            device="cuda:1",
            freeze=False,
            chunk_duration=2.0,
            sample_rate=8000,
        )
        assert cfg.model_size == "small"
        assert cfg.device == "cuda:1"
        assert cfg.freeze is False
        assert cfg.chunk_duration == 2.0
        assert cfg.sample_rate == 8000
        assert cfg.encoder_dim == 768


# ============================================================================
# WhisperEncoderFeatures tests
# ============================================================================

class TestWhisperEncoderFeatures:
    """Tests for WhisperEncoderFeatures (with mocked Whisper backend)."""

    def test_init_does_not_load_model(self):
        """Construction should be lazy -- no model loaded yet."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        assert encoder._is_loaded is False
        assert encoder.encoder is None

    def test_encoder_dim_property_without_loading(self):
        """encoder_dim should be available without loading the model."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="small")
        assert encoder.encoder_dim == 768
        assert encoder._is_loaded is False  # Still not loaded

    def test_encoder_dim_property_all_sizes(self):
        """Verify encoder_dim for several model sizes."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        expected = {
            "tiny": 384, "base": 512, "small": 768,
            "medium": 1024, "large": 1280, "large-v2": 1280, "large-v3": 1280,
        }
        for size, dim in expected.items():
            encoder = WhisperEncoderFeatures(model_size=size)
            assert encoder.encoder_dim == dim, f"Failed for {size}"

    def test_extract_features_mean_pool(self):
        """Mean pooling should produce (batch, encoder_dim)."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        fake_enc = _make_fake_encoder(encoder_dim=384, total_frames=1500)
        encoder.encoder = fake_enc
        encoder._is_loaded = True

        # Fake _audio_to_mel to avoid needing whisper
        batch, n_mels, n_mel_frames = 2, 80, 3000
        with patch.object(encoder, "_audio_to_mel", return_value=torch.randn(batch, n_mels, n_mel_frames)):
            result = encoder.extract_features(
                np.random.randn(batch, 16000).astype(np.float32),
                pool="mean",
                chunk_duration=1.0,
            )
        assert result.shape == (batch, 384)

    def test_extract_features_max_pool(self):
        """Max pooling should produce (batch, encoder_dim)."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        fake_enc = _make_fake_encoder(encoder_dim=384, total_frames=1500)
        encoder.encoder = fake_enc
        encoder._is_loaded = True

        batch = 3
        with patch.object(encoder, "_audio_to_mel", return_value=torch.randn(batch, 80, 3000)):
            result = encoder.extract_features(
                np.random.randn(batch, 16000).astype(np.float32),
                pool="max",
                chunk_duration=1.0,
            )
        assert result.shape == (batch, 384)

    def test_extract_features_first_pool(self):
        """'first' pooling should take the first frame."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="base", freeze=True)
        fake_enc = _make_fake_encoder(encoder_dim=512, total_frames=1500)
        encoder.encoder = fake_enc
        encoder._is_loaded = True

        batch = 1
        with patch.object(encoder, "_audio_to_mel", return_value=torch.randn(batch, 80, 3000)):
            result = encoder.extract_features(
                np.random.randn(batch, 16000).astype(np.float32),
                pool="first",
                chunk_duration=1.0,
            )
        assert result.shape == (batch, 512)

    def test_extract_features_last_pool(self):
        """'last' pooling should take the last content frame."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="base", freeze=True)
        fake_enc = _make_fake_encoder(encoder_dim=512, total_frames=1500)
        encoder.encoder = fake_enc
        encoder._is_loaded = True

        batch = 2
        with patch.object(encoder, "_audio_to_mel", return_value=torch.randn(batch, 80, 3000)):
            result = encoder.extract_features(
                np.random.randn(batch, 16000).astype(np.float32),
                pool="last",
                chunk_duration=1.0,
            )
        assert result.shape == (batch, 512)

    def test_extract_features_no_pool(self):
        """'none' pooling should return full sequence (batch, frames, dim)."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="small", freeze=True)
        fake_enc = _make_fake_encoder(encoder_dim=768, total_frames=1500)
        encoder.encoder = fake_enc
        encoder._is_loaded = True

        batch = 2
        with patch.object(encoder, "_audio_to_mel", return_value=torch.randn(batch, 80, 3000)):
            result = encoder.extract_features(
                np.random.randn(batch, 16000).astype(np.float32),
                pool="none",
                chunk_duration=1.0,
            )
        assert result.shape == (batch, 1500, 768)

    def test_extract_features_invalid_pool_raises(self):
        """An unsupported pool mode should raise ValueError."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        fake_enc = _make_fake_encoder(encoder_dim=384, total_frames=1500)
        encoder.encoder = fake_enc
        encoder._is_loaded = True

        with patch.object(encoder, "_audio_to_mel", return_value=torch.randn(1, 80, 3000)):
            with pytest.raises(ValueError, match="Unknown pooling strategy"):
                encoder.extract_features(
                    np.random.randn(1, 16000).astype(np.float32),
                    pool="attention",
                    chunk_duration=1.0,
                )

    def test_content_frames_formula(self):
        """Verify the content_frames calculation: int(total_frames * chunk_duration / 30.0).

        For 1s chunk and 1500 total frames: int(1500 * 1.0 / 30.0) = 50.
        """
        total_frames = 1500
        chunk_duration = 1.0
        content_frames = max(1, int(total_frames * chunk_duration / 30.0))
        assert content_frames == 50

        # 5-second chunk -> 250 frames
        content_frames_5s = max(1, int(total_frames * 5.0 / 30.0))
        assert content_frames_5s == 250

        # 30-second chunk -> all 1500 frames
        content_frames_30s = max(1, int(total_frames * 30.0 / 30.0))
        assert content_frames_30s == 1500

        # Very short chunk (0.1s) -> 5 frames
        content_frames_short = max(1, int(total_frames * 0.1 / 30.0))
        assert content_frames_short == 5

    def test_to_method_returns_self(self):
        """to('cpu') should return the same WhisperEncoderFeatures instance."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        result = encoder.to("cpu")
        assert result is encoder

    def test_load_state_dict_triggers_load_when_encoder_keys_present(self):
        """load_state_dict should call _load_model when state dict has encoder keys."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        fake_state = {"encoder.weight": torch.randn(10, 10)}

        with patch.object(encoder, "_load_model") as mock_load:
            mock_load.side_effect = lambda: (
                setattr(encoder, "_is_loaded", True)
                or setattr(encoder, "encoder", nn.Linear(10, 10))
            )
            try:
                encoder.load_state_dict(fake_state, strict=False)
            except Exception:
                pass  # shape mismatch is expected
            mock_load.assert_called_once()

    def test_load_state_dict_skips_load_without_encoder_keys(self):
        """load_state_dict should NOT call _load_model when no encoder keys."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        fake_state = {"other_module.weight": torch.randn(5)}

        with patch.object(encoder, "_load_model") as mock_load:
            try:
                encoder.load_state_dict(fake_state, strict=False)
            except Exception:
                pass
            mock_load.assert_not_called()

    def test_forward_is_alias_for_extract_features(self):
        """forward() should delegate to extract_features()."""
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        encoder = WhisperEncoderFeatures(model_size="tiny", freeze=True)
        fake_enc = _make_fake_encoder(encoder_dim=384, total_frames=1500)
        encoder.encoder = fake_enc
        encoder._is_loaded = True

        audio = np.random.randn(1, 16000).astype(np.float32)

        with patch.object(encoder, "_audio_to_mel", return_value=torch.randn(1, 80, 3000)):
            via_forward = encoder.forward(audio, pool="mean", chunk_duration=1.0)
            # Shape should be (1, 384) regardless of path
            assert via_forward.shape == (1, 384)


# ============================================================================
# AudioClassifierConfig tests
# ============================================================================

class TestAudioClassifierConfig:
    """Tests for AudioClassifierConfig dataclass."""

    def test_default_values(self):
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        cfg = AudioClassifierConfig()
        assert cfg.whisper_model_size == "large-v3"
        assert cfg.freeze_encoder is True
        assert cfg.hidden_dim == 256
        assert cfg.dropout == 0.3
        assert cfg.predict_vocalization is True
        assert cfg.predict_energy is True
        assert cfg.sample_rate == 16000
        assert cfg.chunk_duration == 1.0

    def test_default_event_classes(self):
        from synchronai.models.audio.audio_classifier import (
            AudioClassifierConfig,
            AUDIO_EVENT_CLASSES,
        )

        cfg = AudioClassifierConfig()
        assert cfg.event_classes == AUDIO_EVENT_CLASSES
        assert len(cfg.event_classes) == 7
        assert "speech" in cfg.event_classes
        assert "silence" in cfg.event_classes

    def test_event_classes_are_independent_copies(self):
        """Each config should get its own copy of the event classes list."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        cfg1 = AudioClassifierConfig()
        cfg2 = AudioClassifierConfig()
        cfg1.event_classes.append("extra")
        assert "extra" not in cfg2.event_classes

    def test_num_event_classes_property(self):
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        cfg = AudioClassifierConfig()
        assert cfg.num_event_classes == 7

        cfg2 = AudioClassifierConfig(event_classes=["a", "b", "c"])
        assert cfg2.num_event_classes == 3

    @pytest.mark.parametrize(
        "model_size, expected_dim",
        [
            ("tiny", 384),
            ("base", 512),
            ("small", 768),
            ("medium", 1024),
            ("large", 1280),
            ("large-v2", 1280),
            ("large-v3", 1280),
        ],
    )
    def test_encoder_dim_property(self, model_size, expected_dim):
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        cfg = AudioClassifierConfig(whisper_model_size=model_size)
        assert cfg.encoder_dim == expected_dim

    def test_asdict_roundtrip(self):
        """Config should survive a dict roundtrip."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        original = AudioClassifierConfig(hidden_dim=512, dropout=0.5)
        d = asdict(original)
        restored = AudioClassifierConfig(**d)
        assert restored.hidden_dim == 512
        assert restored.dropout == 0.5
        assert restored.event_classes == original.event_classes


# ============================================================================
# AudioClassifier tests
# ============================================================================

class TestAudioClassifier:
    """Tests for AudioClassifier nn.Module (with mocked encoder)."""

    def test_forward_output_keys_all_heads(self):
        """Forward should return all expected keys when all heads are enabled."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            predict_vocalization=True,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 4
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                outputs = model(audio)

        expected_keys = {
            "event_logits", "event_probs", "features", "hidden_features",
            "vocalization_logit", "vocalization_prob", "energy",
        }
        assert set(outputs.keys()) == expected_keys

    def test_forward_output_shapes(self):
        """Verify tensor shapes for each output."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            hidden_dim=256,
            predict_vocalization=True,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 4
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        assert out["event_logits"].shape == (batch, 7)
        assert out["event_probs"].shape == (batch, 7)
        assert out["features"].shape == (batch, 1280)
        assert out["hidden_features"].shape == (batch, 256)
        assert out["vocalization_logit"].shape == (batch, 1)
        assert out["vocalization_prob"].shape == (batch, 1)
        assert out["energy"].shape == (batch, 1)

    def test_event_probs_sum_to_one(self):
        """Softmax event probabilities should sum to ~1 along class dim."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3")
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 2
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        sums = out["event_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch), atol=1e-5)

    def test_vocalization_prob_in_zero_one(self):
        """Vocalization probability (sigmoid output) should be in [0, 1]."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3", predict_vocalization=True)
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 3
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        assert (out["vocalization_prob"] >= 0).all()
        assert (out["vocalization_prob"] <= 1).all()

    def test_forward_without_vocalization_head(self):
        """When predict_vocalization=False, output should lack vocalization keys."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            predict_vocalization=False,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 2
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        assert "vocalization_logit" not in out
        assert "vocalization_prob" not in out
        assert "energy" in out
        assert model.vocalization_head is None

    def test_forward_without_energy_head(self):
        """When predict_energy=False, output should lack energy key."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            predict_vocalization=True,
            predict_energy=False,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 2
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        assert "energy" not in out
        assert "vocalization_logit" in out
        assert model.energy_head is None

    def test_forward_without_both_optional_heads(self):
        """When both optional heads are off, only event + feature keys remain."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            predict_vocalization=False,
            predict_energy=False,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 2
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        expected_keys = {"event_logits", "event_probs", "features", "hidden_features"}
        assert set(out.keys()) == expected_keys

    def test_forward_with_different_model_sizes(self):
        """Verify that different model sizes produce correct feature dimensions."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        for size, expected_dim in [("tiny", 384), ("small", 768), ("medium", 1024)]:
            config = AudioClassifierConfig(whisper_model_size=size)
            model = _build_classifier_with_fake_encoder(config)
            model.eval()

            batch = 2
            n_mels = 80
            audio = torch.randn(batch, 16000)

            with patch.object(
                model.encoder, "_audio_to_mel",
                return_value=torch.randn(batch, n_mels, 3000),
            ):
                with torch.no_grad():
                    out = model(audio)

            assert out["features"].shape == (batch, expected_dim), (
                f"Model size '{size}' should produce features of dim {expected_dim}"
            )


# ============================================================================
# Freeze / unfreeze tests
# ============================================================================

class TestFreezeUnfreeze:
    """Tests for freeze_encoder() and unfreeze_encoder() methods."""

    def test_freeze_encoder_sets_requires_grad_false(self):
        """freeze_encoder should set requires_grad=False on all encoder params."""
        model = _build_classifier_with_fake_encoder()

        # First unfreeze to ensure params start with requires_grad=True
        model.unfreeze_encoder()
        for p in model.encoder.parameters():
            assert p.requires_grad is True

        model.freeze_encoder()
        for p in model.encoder.parameters():
            assert p.requires_grad is False

    def test_unfreeze_encoder_sets_requires_grad_true(self):
        """unfreeze_encoder should set requires_grad=True on all encoder params."""
        model = _build_classifier_with_fake_encoder()

        model.freeze_encoder()
        for p in model.encoder.parameters():
            assert p.requires_grad is False

        model.unfreeze_encoder()
        for p in model.encoder.parameters():
            assert p.requires_grad is True

    def test_freeze_does_not_affect_heads(self):
        """Freezing/unfreezing the encoder should not change head parameters."""
        model = _build_classifier_with_fake_encoder()

        # Heads should always have requires_grad=True
        model.freeze_encoder()
        for p in model.event_head.parameters():
            assert p.requires_grad is True
        for p in model.feature_proj.parameters():
            assert p.requires_grad is True

        model.unfreeze_encoder()
        for p in model.event_head.parameters():
            assert p.requires_grad is True
        for p in model.feature_proj.parameters():
            assert p.requires_grad is True

    def test_multiple_freeze_unfreeze_cycles(self):
        """Repeated freeze/unfreeze cycles should work without issues."""
        model = _build_classifier_with_fake_encoder()

        for _ in range(3):
            model.freeze_encoder()
            assert all(not p.requires_grad for p in model.encoder.parameters())
            model.unfreeze_encoder()
            assert all(p.requires_grad for p in model.encoder.parameters())


# ============================================================================
# get_parameter_groups tests
# ============================================================================

class TestGetParameterGroups:
    """Tests for get_parameter_groups() with different learning rates."""

    def test_frozen_encoder_returns_empty_backbone_group(self):
        """When encoder is frozen, no encoder group should appear."""
        model = _build_classifier_with_fake_encoder()
        model.freeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        # No group should have lr == encoder_lr because encoder is frozen
        for group in groups:
            assert group["lr"] != 1e-5, "Frozen encoder should not appear in param groups"

        # Should still have head groups
        assert len(groups) >= 1

    def test_unfrozen_encoder_includes_backbone_group(self):
        """When encoder is unfrozen, encoder group should be present."""
        model = _build_classifier_with_fake_encoder()
        model.unfreeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        encoder_groups = [g for g in groups if g["lr"] == 1e-5]
        assert len(encoder_groups) == 1, "Should have exactly one encoder group"

        # The encoder group should have parameters
        encoder_params = list(encoder_groups[0]["params"])
        assert len(encoder_params) > 0

    def test_head_groups_always_present(self):
        """Head parameter groups (feature_proj, event_head, etc.) should always be present."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            predict_vocalization=True,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.freeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)
        head_groups = [g for g in groups if g["lr"] == 1e-3]

        # Should have: feature_proj, event_head, vocalization_head, energy_head
        assert len(head_groups) == 4

    def test_head_groups_without_optional_heads(self):
        """Without optional heads, fewer head groups should be returned."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            predict_vocalization=False,
            predict_energy=False,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.freeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)
        head_groups = [g for g in groups if g["lr"] == 1e-3]

        # Should have only: feature_proj, event_head
        assert len(head_groups) == 2

    def test_all_trainable_params_are_covered(self):
        """Every parameter with requires_grad=True should be in some group."""
        model = _build_classifier_with_fake_encoder()
        model.unfreeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        grouped_params = set()
        for group in groups:
            for p in group["params"]:
                grouped_params.add(id(p))

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert id(p) in grouped_params, (
                    f"Parameter '{name}' requires_grad but is not in any group"
                )

    def test_parameter_groups_can_be_passed_to_optimizer(self):
        """Parameter groups should be valid input for torch.optim.Adam."""
        model = _build_classifier_with_fake_encoder()
        model.unfreeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        # Should not raise
        optimizer = torch.optim.Adam(groups)
        assert optimizer is not None


# ============================================================================
# predict_event tests
# ============================================================================

class TestPredictEvent:
    """Tests for predict_event() convenience method."""

    def test_returns_tuple_of_str_and_float(self):
        """predict_event should return (event_class_str, confidence_float)."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3")
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        audio = torch.randn(1, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(1, 128, 3000)):
            event_class, confidence = model.predict_event(audio)

        assert isinstance(event_class, str)
        assert isinstance(confidence, float)

    def test_event_class_is_valid(self):
        """Predicted class should be one of the configured event classes."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3")
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        audio = torch.randn(1, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(1, 128, 3000)):
            event_class, _ = model.predict_event(audio)

        assert event_class in config.event_classes

    def test_confidence_is_between_zero_and_one(self):
        """Confidence (from softmax) should be in [0, 1]."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3")
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        audio = torch.randn(1, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(1, 128, 3000)):
            _, confidence = model.predict_event(audio)

        assert 0.0 <= confidence <= 1.0

    def test_predict_event_with_batched_input(self):
        """predict_event should handle batched input (takes first element)."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3")
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        audio = torch.randn(4, 16000)  # batch of 4

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(4, 128, 3000)):
            event_class, confidence = model.predict_event(audio)

        assert isinstance(event_class, str)
        assert event_class in config.event_classes
        assert 0.0 <= confidence <= 1.0

    def test_predict_event_with_numpy_input(self):
        """predict_event should accept numpy arrays."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3")
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        audio = np.random.randn(1, 16000).astype(np.float32)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(1, 128, 3000)):
            event_class, confidence = model.predict_event(audio)

        assert isinstance(event_class, str)
        assert isinstance(confidence, float)


# ============================================================================
# WHISPER_DIMS constant tests
# ============================================================================

class TestWhisperDims:
    """Tests for the WHISPER_DIMS mapping."""

    def test_all_expected_sizes_present(self):
        from synchronai.models.audio.whisper_encoder import WHISPER_DIMS

        expected_sizes = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
        assert expected_sizes == set(WHISPER_DIMS.keys())

    def test_dims_are_positive_integers(self):
        from synchronai.models.audio.whisper_encoder import WHISPER_DIMS

        for size, dim in WHISPER_DIMS.items():
            assert isinstance(dim, int), f"Dim for {size} should be int"
            assert dim > 0, f"Dim for {size} should be positive"

    def test_dims_increase_with_model_size(self):
        """Larger models should have larger (or equal) encoder dimensions."""
        from synchronai.models.audio.whisper_encoder import WHISPER_DIMS

        ordered = ["tiny", "base", "small", "medium", "large"]
        for i in range(len(ordered) - 1):
            assert WHISPER_DIMS[ordered[i]] <= WHISPER_DIMS[ordered[i + 1]]


# ============================================================================
# AUDIO_EVENT_CLASSES constant tests
# ============================================================================

class TestAudioEventClasses:
    """Tests for the AUDIO_EVENT_CLASSES constant."""

    def test_seven_classes(self):
        from synchronai.models.audio.audio_classifier import AUDIO_EVENT_CLASSES

        assert len(AUDIO_EVENT_CLASSES) == 7

    def test_expected_classes(self):
        from synchronai.models.audio.audio_classifier import AUDIO_EVENT_CLASSES

        expected = {"speech", "laughter", "crying", "babbling", "silence", "noise", "music"}
        assert set(AUDIO_EVENT_CLASSES) == expected

    def test_classes_are_strings(self):
        from synchronai.models.audio.audio_classifier import AUDIO_EVENT_CLASSES

        for cls in AUDIO_EVENT_CLASSES:
            assert isinstance(cls, str)


# ============================================================================
# build_audio_classifier helper tests
# ============================================================================

class TestBuildAudioClassifier:
    """Tests for the build_audio_classifier factory function."""

    def test_returns_audio_classifier_instance(self):
        from synchronai.models.audio.audio_classifier import (
            AudioClassifier,
            AudioClassifierConfig,
            build_audio_classifier,
        )

        with patch(
            "synchronai.models.audio.whisper_encoder.WhisperEncoderFeatures._load_model",
        ):
            config = AudioClassifierConfig(whisper_model_size="large-v3")
            model = build_audio_classifier(config)
            assert isinstance(model, AudioClassifier)

    def test_config_stored_on_model(self):
        from synchronai.models.audio.audio_classifier import (
            AudioClassifierConfig,
            build_audio_classifier,
        )

        with patch(
            "synchronai.models.audio.whisper_encoder.WhisperEncoderFeatures._load_model",
        ):
            config = AudioClassifierConfig(hidden_dim=512)
            model = build_audio_classifier(config)
            assert model.config is config
            assert model.config.hidden_dim == 512


# ============================================================================
# get_whisper_encoder helper tests
# ============================================================================

class TestGetWhisperEncoder:
    """Tests for get_whisper_encoder factory and caching."""

    def test_returns_whisper_encoder_features(self):
        from synchronai.models.audio.whisper_encoder import (
            WhisperEncoderFeatures,
            get_whisper_encoder,
            clear_whisper_cache,
        )

        clear_whisper_cache()
        encoder = get_whisper_encoder(model_size="tiny", use_cache=False)
        assert isinstance(encoder, WhisperEncoderFeatures)

    def test_caching_returns_same_instance(self):
        from synchronai.models.audio.whisper_encoder import (
            get_whisper_encoder,
            clear_whisper_cache,
        )

        clear_whisper_cache()
        enc1 = get_whisper_encoder(model_size="tiny", use_cache=True)
        enc2 = get_whisper_encoder(model_size="tiny", use_cache=True)
        assert enc1 is enc2

    def test_different_config_creates_new_instance(self):
        from synchronai.models.audio.whisper_encoder import (
            get_whisper_encoder,
            clear_whisper_cache,
        )

        clear_whisper_cache()
        enc1 = get_whisper_encoder(model_size="tiny", freeze=True, use_cache=True)
        enc2 = get_whisper_encoder(model_size="tiny", freeze=False, use_cache=True)
        assert enc1 is not enc2

    def test_clear_cache_resets(self):
        from synchronai.models.audio.whisper_encoder import (
            get_whisper_encoder,
            clear_whisper_cache,
        )

        clear_whisper_cache()
        enc1 = get_whisper_encoder(model_size="tiny", use_cache=True)
        clear_whisper_cache()
        enc2 = get_whisper_encoder(model_size="tiny", use_cache=True)
        assert enc1 is not enc2


# ============================================================================
# Edge case and integration-style tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_single_sample_unbatched_shape(self):
        """A 1D audio tensor should still work through forward (the encoder handles it)."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(whisper_model_size="large-v3")
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        # Single unbatched sample -- _audio_to_mel in real code adds batch dim
        audio = torch.randn(16000)

        # Mock _audio_to_mel to return batched mel (as the real version does)
        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(1, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        assert out["event_logits"].shape == (1, 7)

    def test_custom_event_classes(self):
        """Model should work with a custom (non-default) set of event classes."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        custom_classes = ["cat_meow", "dog_bark", "bird_chirp"]
        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            event_classes=custom_classes,
        )
        model = _build_classifier_with_fake_encoder(config)
        model.eval()

        batch = 2
        audio = torch.randn(batch, 16000)

        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(batch, 128, 3000)):
            with torch.no_grad():
                out = model(audio)

        assert out["event_logits"].shape == (batch, 3)
        assert out["event_probs"].shape == (batch, 3)

        # predict_event should return one of the custom classes
        with patch.object(model.encoder, "_audio_to_mel", return_value=torch.randn(1, 128, 3000)):
            event_class, _ = model.predict_event(audio[:1])
        assert event_class in custom_classes

    def test_model_is_nn_module(self):
        """AudioClassifier should be a proper nn.Module."""
        from synchronai.models.audio.audio_classifier import AudioClassifier

        model = _build_classifier_with_fake_encoder()
        assert isinstance(model, nn.Module)
        assert isinstance(model, AudioClassifier)

    def test_feature_proj_structure(self):
        """feature_proj should be a Sequential with expected layer types."""
        model = _build_classifier_with_fake_encoder()

        assert isinstance(model.feature_proj, nn.Sequential)

        layer_types = [type(layer) for layer in model.feature_proj]
        # Expected: Linear, BatchNorm1d, ReLU, Dropout, Linear, BatchNorm1d, ReLU, Dropout
        assert layer_types == [
            nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout,
            nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout,
        ]

    def test_feature_proj_dimensions(self):
        """feature_proj first Linear should take encoder_dim, last should output hidden_dim."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            whisper_model_size="large-v3",
            hidden_dim=256,
        )
        model = _build_classifier_with_fake_encoder(config)

        first_linear = model.feature_proj[0]
        last_linear = model.feature_proj[4]  # Second Linear (index 4 in the Sequential)

        assert first_linear.in_features == 1280  # large-v3 encoder dim
        assert last_linear.out_features == 256  # hidden_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

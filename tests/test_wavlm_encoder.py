"""
Comprehensive tests for the WavLM audio encoder and its integration with AudioClassifier.

Tests cover:
- WavLMEncoderConfig: dataclass fields, encoder_dim/num_layers properties, short name aliases
- WavLMEncoderFeatures: lazy loading, extract_features pooling modes, layer_weights,
  freeze/unfreeze, get_parameter_groups, load_state_dict triggers
- AudioClassifier integration: forward pass with WavLM backend, output shapes/keys,
  freeze/unfreeze, get_parameter_groups with layer_weights, predict_event
- Backward compatibility: Whisper config still works alongside WavLM
- WAVLM_CONFIGS constant validation
- get_wavlm_encoder factory and caching

The WavLM HuggingFace model is mocked throughout to avoid requiring the
transformers package or downloading model weights.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_fake_wavlm(hidden_size: int = 1024, num_layers: int = 24, seq_len: int = 50):
    """Create a fake nn.Module that mimics HuggingFace WavLM output format.

    When called with ``(audio, output_hidden_states=True)`` the fake model
    returns an object whose ``.hidden_states`` attribute is a tuple of
    ``(num_layers + 1)`` tensors, each of shape ``(batch, seq_len, hidden_size)``.
    """

    class _FakeOutput:
        """Mimics the HuggingFace BaseModelOutput with hidden_states."""

        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class FakeWavLM(nn.Module):
        def __init__(self):
            super().__init__()
            # A real parameter so that parameters() is non-empty and
            # freeze/unfreeze can be tested.
            self.proj = nn.Linear(hidden_size, hidden_size)
            self._hidden_size = hidden_size
            self._num_layers = num_layers
            self._seq_len = seq_len

        def forward(self, audio, output_hidden_states=False):
            batch = audio.shape[0]
            # Produce num_layers + 1 hidden state tensors (embedding + transformer layers)
            hidden_states = tuple(
                torch.randn(batch, self._seq_len, self._hidden_size)
                for _ in range(self._num_layers + 1)
            )
            return _FakeOutput(hidden_states=hidden_states)

    return FakeWavLM()


def _build_classifier_with_fake_wavlm(config=None, hidden_size=1024, num_layers=24, seq_len=50):
    """Build an AudioClassifier with encoder_type='wavlm' using a fake WavLM model.

    Patches ``WavLMEncoderFeatures._load_model`` to inject a fake WavLM module
    so that no real model weights are downloaded.
    """
    from synchronai.models.audio.audio_classifier import AudioClassifier, AudioClassifierConfig

    if config is None:
        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
        )

    # Determine the expected hidden_size from the config's wavlm_model_name
    from synchronai.models.audio.wavlm_encoder import WAVLM_CONFIGS, _resolve_model_name

    resolved = _resolve_model_name(config.wavlm_model_name)
    model_cfg = WAVLM_CONFIGS.get(resolved, WAVLM_CONFIGS.get(config.wavlm_model_name))
    if model_cfg:
        hidden_size = model_cfg["hidden_size"]
        num_layers = model_cfg["num_layers"]

    with patch(
        "synchronai.models.audio.wavlm_encoder.WavLMEncoderFeatures._load_model",
        autospec=True,
    ) as mock_load:
        def side_effect(self_arg):
            self_arg.wavlm = _make_fake_wavlm(hidden_size, num_layers, seq_len)
            self_arg._is_loaded = True

        mock_load.side_effect = side_effect

        model = AudioClassifier(config)

        # Trigger the fake load so the encoder is populated
        model.encoder._load_model()

    return model


@pytest.fixture(autouse=True)
def _clear_wavlm_cache():
    """Clear the global WavLM encoder cache before and after each test."""
    from synchronai.models.audio.wavlm_encoder import clear_wavlm_cache

    clear_wavlm_cache()
    yield
    clear_wavlm_cache()


# ============================================================================
# WavLMEncoderConfig tests
# ============================================================================

class TestWavLMEncoderConfig:
    """Tests for WavLMEncoderConfig dataclass and its properties."""

    def test_default_values(self):
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderConfig

        cfg = WavLMEncoderConfig()
        assert cfg.model_name == "microsoft/wavlm-large"
        assert cfg.device is None
        assert cfg.freeze is True
        assert cfg.chunk_duration == 1.0
        assert cfg.sample_rate == 16000

    @pytest.mark.parametrize(
        "model_name, expected_dim",
        [
            ("microsoft/wavlm-large", 1024),
            ("microsoft/wavlm-base-plus", 768),
            ("microsoft/wavlm-base", 768),
            # Short name aliases
            ("wavlm-large", 1024),
            ("wavlm-base-plus", 768),
            ("wavlm-base", 768),
        ],
    )
    def test_encoder_dim_property(self, model_name, expected_dim):
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderConfig

        cfg = WavLMEncoderConfig(model_name=model_name)
        assert cfg.encoder_dim == expected_dim

    @pytest.mark.parametrize(
        "model_name, expected_layers",
        [
            ("microsoft/wavlm-large", 24),
            ("microsoft/wavlm-base-plus", 12),
            ("microsoft/wavlm-base", 12),
            ("wavlm-large", 24),
            ("wavlm-base-plus", 12),
        ],
    )
    def test_num_layers_property(self, model_name, expected_layers):
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderConfig

        cfg = WavLMEncoderConfig(model_name=model_name)
        assert cfg.num_layers == expected_layers

    def test_short_name_aliases(self):
        """Short aliases should resolve to the same dims as full HuggingFace names."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderConfig

        short = WavLMEncoderConfig(model_name="wavlm-large")
        full = WavLMEncoderConfig(model_name="microsoft/wavlm-large")
        assert short.encoder_dim == full.encoder_dim
        assert short.num_layers == full.num_layers

    def test_unknown_model_defaults_to_large(self):
        """Unknown model names should fall back to wavlm-large defaults."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderConfig

        cfg = WavLMEncoderConfig(model_name="some-unknown-model")
        assert cfg.encoder_dim == 1024
        assert cfg.num_layers == 24

    def test_custom_values(self):
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderConfig

        cfg = WavLMEncoderConfig(
            model_name="wavlm-base",
            device="cuda:0",
            freeze=False,
            chunk_duration=2.0,
            sample_rate=8000,
        )
        assert cfg.device == "cuda:0"
        assert cfg.freeze is False
        assert cfg.chunk_duration == 2.0
        assert cfg.sample_rate == 8000
        assert cfg.encoder_dim == 768


# ============================================================================
# WavLMEncoderFeatures tests
# ============================================================================

class TestWavLMEncoderFeatures:
    """Tests for WavLMEncoderFeatures (with mocked WavLM backend)."""

    def test_init_does_not_load_model(self):
        """Construction should be lazy -- no model loaded yet."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="wavlm-large", freeze=True)
        assert encoder._is_loaded is False
        assert encoder.wavlm is None

    def test_encoder_dim_property_without_loading(self):
        """encoder_dim should be available without loading the model."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="microsoft/wavlm-large")
        assert encoder.encoder_dim == 1024
        assert encoder._is_loaded is False

    def _make_encoder_with_fake_wavlm(
        self,
        model_name="microsoft/wavlm-large",
        hidden_size=1024,
        num_layers=24,
        seq_len=50,
        freeze=True,
    ):
        """Helper: create a WavLMEncoderFeatures with a fake WavLM injected."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name=model_name, freeze=freeze)
        encoder.wavlm = _make_fake_wavlm(hidden_size, num_layers, seq_len)
        encoder._is_loaded = True
        return encoder

    def test_extract_features_mean_pool(self):
        """Mean pooling should produce (batch, encoder_dim)."""
        encoder = self._make_encoder_with_fake_wavlm()
        batch = 2
        audio = torch.randn(batch, 16000)
        result = encoder.extract_features(audio, pool="mean", chunk_duration=1.0)
        assert result.shape == (batch, 1024)

    def test_extract_features_max_pool(self):
        """Max pooling should produce (batch, encoder_dim)."""
        encoder = self._make_encoder_with_fake_wavlm()
        batch = 3
        audio = torch.randn(batch, 16000)
        result = encoder.extract_features(audio, pool="max", chunk_duration=1.0)
        assert result.shape == (batch, 1024)

    def test_extract_features_first_pool(self):
        """'first' pooling should take the first frame -> (batch, encoder_dim)."""
        encoder = self._make_encoder_with_fake_wavlm(
            model_name="microsoft/wavlm-base-plus",
            hidden_size=768,
            num_layers=12,
        )
        batch = 1
        audio = torch.randn(batch, 16000)
        result = encoder.extract_features(audio, pool="first", chunk_duration=1.0)
        assert result.shape == (batch, 768)

    def test_extract_features_last_pool(self):
        """'last' pooling should take the last content frame."""
        encoder = self._make_encoder_with_fake_wavlm(
            model_name="microsoft/wavlm-base-plus",
            hidden_size=768,
            num_layers=12,
        )
        batch = 2
        audio = torch.randn(batch, 16000)
        result = encoder.extract_features(audio, pool="last", chunk_duration=1.0)
        assert result.shape == (batch, 768)

    def test_extract_features_no_pool(self):
        """'none' pooling should return full sequence (batch, seq_len, encoder_dim)."""
        seq_len = 50
        encoder = self._make_encoder_with_fake_wavlm(seq_len=seq_len)
        batch = 2
        audio = torch.randn(batch, 16000)
        result = encoder.extract_features(audio, pool="none", chunk_duration=1.0)
        assert result.shape == (batch, seq_len, 1024)

    def test_extract_features_invalid_pool_raises(self):
        """An unsupported pool mode should raise ValueError."""
        encoder = self._make_encoder_with_fake_wavlm()
        audio = torch.randn(1, 16000)
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            encoder.extract_features(audio, pool="attention", chunk_duration=1.0)

    def test_layer_weights_initialized_uniform(self):
        """layer_weights should be initialized to uniform values (all equal)."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="microsoft/wavlm-large")
        weights = encoder.layer_weights.data
        # All elements should be equal (uniform initialization)
        assert torch.allclose(weights, weights[0].expand_as(weights))
        # And they should be 1/(num_layers+1)
        expected_val = 1.0 / 25.0
        assert torch.allclose(weights, torch.full_like(weights, expected_val))

    def test_layer_weights_count_matches_num_layers_plus_one(self):
        """layer_weights size should be num_layers + 1 for each model variant."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        # wavlm-large: 24 layers -> 25 weights
        encoder_large = WavLMEncoderFeatures(model_name="microsoft/wavlm-large")
        assert encoder_large.layer_weights.shape == (25,)

        # wavlm-base-plus: 12 layers -> 13 weights
        encoder_base = WavLMEncoderFeatures(model_name="microsoft/wavlm-base-plus")
        assert encoder_base.layer_weights.shape == (13,)

    def test_forward_is_alias_for_extract_features(self):
        """forward() should delegate to extract_features()."""
        encoder = self._make_encoder_with_fake_wavlm()
        audio = torch.randn(1, 16000)

        via_forward = encoder.forward(audio, pool="mean", chunk_duration=1.0)
        assert via_forward.shape == (1, 1024)

    def test_to_method_returns_self(self):
        """to('cpu') should return the same WavLMEncoderFeatures instance."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="wavlm-large", freeze=True)
        result = encoder.to("cpu")
        assert result is encoder

    def test_load_state_dict_triggers_load_when_wavlm_keys_present(self):
        """load_state_dict should call _load_model when state dict has wavlm.* keys."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="wavlm-large", freeze=True)
        fake_state = {"wavlm.encoder.weight": torch.randn(10, 10)}

        with patch.object(encoder, "_load_model") as mock_load:
            mock_load.side_effect = lambda: (
                setattr(encoder, "_is_loaded", True)
                or setattr(encoder, "wavlm", nn.Linear(10, 10))
            )
            try:
                encoder.load_state_dict(fake_state, strict=False)
            except Exception:
                pass  # shape mismatch is expected
            mock_load.assert_called_once()

    def test_load_state_dict_skips_load_without_wavlm_keys(self):
        """load_state_dict should NOT call _load_model when no wavlm.* keys."""
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="wavlm-large", freeze=True)
        fake_state = {"other_module.weight": torch.randn(5)}

        with patch.object(encoder, "_load_model") as mock_load:
            try:
                encoder.load_state_dict(fake_state, strict=False)
            except Exception:
                pass
            mock_load.assert_not_called()

    def test_extract_features_with_numpy_input(self):
        """extract_features should accept numpy arrays."""
        encoder = self._make_encoder_with_fake_wavlm()
        audio = np.random.randn(2, 16000).astype(np.float32)
        result = encoder.extract_features(audio, pool="mean", chunk_duration=1.0)
        assert result.shape == (2, 1024)

    def test_extract_features_with_1d_input(self):
        """A single 1D audio tensor should be handled (unsqueezed to batch=1)."""
        encoder = self._make_encoder_with_fake_wavlm()
        audio = torch.randn(16000)
        result = encoder.extract_features(audio, pool="mean", chunk_duration=1.0)
        assert result.shape == (1, 1024)


# ============================================================================
# Freeze / unfreeze tests
# ============================================================================

class TestWavLMFreezeUnfreeze:
    """Tests for freeze_encoder() and unfreeze_encoder() on WavLMEncoderFeatures."""

    def _make_encoder(self, freeze=True):
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="microsoft/wavlm-large", freeze=freeze)
        encoder.wavlm = _make_fake_wavlm(1024, 24, 50)
        encoder._is_loaded = True
        return encoder

    def test_freeze_encoder_freezes_wavlm_params(self):
        """freeze_encoder should set requires_grad=False on all wavlm params."""
        encoder = self._make_encoder(freeze=False)
        # Confirm unfrozen initially
        for p in encoder.wavlm.parameters():
            assert p.requires_grad is True

        encoder.freeze_encoder()
        for p in encoder.wavlm.parameters():
            assert p.requires_grad is False

    def test_freeze_encoder_preserves_layer_weights_trainability(self):
        """Freezing wavlm should NOT freeze layer_weights -- they remain trainable."""
        encoder = self._make_encoder(freeze=False)
        encoder.freeze_encoder()

        assert encoder.layer_weights.requires_grad is True

    def test_unfreeze_encoder_unfreezes_wavlm_params(self):
        """unfreeze_encoder should set requires_grad=True on all wavlm params."""
        encoder = self._make_encoder(freeze=True)
        # Freeze first
        encoder.freeze_encoder()
        for p in encoder.wavlm.parameters():
            assert p.requires_grad is False

        encoder.unfreeze_encoder()
        for p in encoder.wavlm.parameters():
            assert p.requires_grad is True

    def test_multiple_freeze_unfreeze_cycles(self):
        """Repeated freeze/unfreeze cycles should work without issues."""
        encoder = self._make_encoder(freeze=False)

        for _ in range(3):
            encoder.freeze_encoder()
            assert all(not p.requires_grad for p in encoder.wavlm.parameters())
            assert encoder.layer_weights.requires_grad is True

            encoder.unfreeze_encoder()
            assert all(p.requires_grad for p in encoder.wavlm.parameters())
            assert encoder.layer_weights.requires_grad is True


# ============================================================================
# get_parameter_groups tests
# ============================================================================

class TestWavLMGetParameterGroups:
    """Tests for get_parameter_groups() on WavLMEncoderFeatures."""

    def _make_encoder(self, freeze=True):
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        encoder = WavLMEncoderFeatures(model_name="microsoft/wavlm-large", freeze=freeze)
        encoder.wavlm = _make_fake_wavlm(1024, 24, 50)
        encoder._is_loaded = True
        if freeze:
            encoder.freeze_encoder()
        return encoder

    def test_frozen_encoder_still_has_layer_weights_group(self):
        """Even with a frozen encoder, layer_weights should appear in param groups."""
        encoder = self._make_encoder(freeze=True)
        groups = encoder.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        # Should have at least one group (the layer_weights group)
        assert len(groups) >= 1

        # Find the group containing layer_weights
        layer_weights_found = False
        for group in groups:
            for p in group["params"]:
                if p is encoder.layer_weights:
                    layer_weights_found = True
                    assert group["lr"] == 1e-3  # layer_weights use head_lr
        assert layer_weights_found, "layer_weights should always be in param groups"

        # No encoder group should be present (frozen)
        encoder_groups = [g for g in groups if g["lr"] == 1e-5]
        assert len(encoder_groups) == 0, "Frozen encoder should not have encoder group"

    def test_unfrozen_encoder_has_both_groups(self):
        """Unfrozen encoder should have both an encoder group and a layer_weights group."""
        encoder = self._make_encoder(freeze=False)
        groups = encoder.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        # Should have exactly 2 groups: encoder backbone + layer_weights
        assert len(groups) == 2

        encoder_groups = [g for g in groups if g["lr"] == 1e-5]
        head_groups = [g for g in groups if g["lr"] == 1e-3]
        assert len(encoder_groups) == 1
        assert len(head_groups) == 1

        # Encoder group should have wavlm params
        encoder_params = list(encoder_groups[0]["params"])
        assert len(encoder_params) > 0

        # Head group should have layer_weights
        head_params = list(head_groups[0]["params"])
        assert len(head_params) == 1
        assert head_params[0] is encoder.layer_weights

    def test_parameter_groups_can_be_passed_to_optimizer(self):
        """Parameter groups should be valid input for torch.optim.Adam."""
        encoder = self._make_encoder(freeze=False)
        groups = encoder.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        # Should not raise
        optimizer = torch.optim.Adam(groups)
        assert optimizer is not None


# ============================================================================
# AudioClassifier integration tests (WavLM backend)
# ============================================================================

class TestWavLMInAudioClassifier:
    """End-to-end tests with AudioClassifier using WavLM encoder backend."""

    def test_wavlm_classifier_forward_output_keys(self):
        """Forward should return the same output keys as the Whisper version."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
            predict_vocalization=True,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.eval()

        batch = 4
        audio = torch.randn(batch, 16000)

        with torch.no_grad():
            outputs = model(audio)

        expected_keys = {
            "event_logits", "event_probs", "features", "hidden_features",
            "vocalization_logit", "vocalization_prob", "energy",
        }
        assert set(outputs.keys()) == expected_keys

    def test_wavlm_classifier_forward_output_shapes(self):
        """Verify tensor shapes with WavLM dims (1024 for wavlm-large)."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
            hidden_dim=256,
            predict_vocalization=True,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.eval()

        batch = 4
        audio = torch.randn(batch, 16000)

        with torch.no_grad():
            out = model(audio)

        assert out["event_logits"].shape == (batch, 7)
        assert out["event_probs"].shape == (batch, 7)
        assert out["features"].shape == (batch, 1024)  # WavLM large encoder_dim
        assert out["hidden_features"].shape == (batch, 256)
        assert out["vocalization_logit"].shape == (batch, 1)
        assert out["vocalization_prob"].shape == (batch, 1)
        assert out["energy"].shape == (batch, 1)

    def test_wavlm_classifier_forward_output_shapes_base_plus(self):
        """Verify tensor shapes with wavlm-base-plus dims (768)."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-base-plus",
            hidden_dim=128,
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.eval()

        batch = 2
        audio = torch.randn(batch, 16000)

        with torch.no_grad():
            out = model(audio)

        assert out["features"].shape == (batch, 768)  # base-plus encoder_dim
        assert out["hidden_features"].shape == (batch, 128)

    def test_wavlm_classifier_freeze_preserves_layer_weights(self):
        """Freezing the encoder in AudioClassifier should keep layer_weights trainable."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
        )
        model = _build_classifier_with_fake_wavlm(config)

        model.freeze_encoder()

        # WavLM params should be frozen
        for p in model.encoder.wavlm.parameters():
            assert p.requires_grad is False

        # layer_weights should still be trainable
        assert model.encoder.layer_weights.requires_grad is True

        # Head params should still be trainable
        for p in model.event_head.parameters():
            assert p.requires_grad is True
        for p in model.feature_proj.parameters():
            assert p.requires_grad is True

    def test_wavlm_classifier_get_parameter_groups(self):
        """get_parameter_groups should include layer_weights even when encoder is frozen."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
            predict_vocalization=True,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.freeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        # No encoder backbone group (frozen), but layer_weights group should exist
        encoder_groups = [g for g in groups if g["lr"] == 1e-5]
        assert len(encoder_groups) == 0, "Frozen encoder backbone should not appear"

        head_groups = [g for g in groups if g["lr"] == 1e-3]
        # Should include: layer_weights, feature_proj, event_head, vocalization_head, energy_head
        assert len(head_groups) == 5

        # Verify layer_weights is in one of those groups
        layer_weights_found = False
        for group in head_groups:
            for p in group["params"]:
                if p is model.encoder.layer_weights:
                    layer_weights_found = True
        assert layer_weights_found, "layer_weights must be in parameter groups"

    def test_wavlm_classifier_get_parameter_groups_unfrozen(self):
        """When encoder is unfrozen, encoder backbone group should also appear."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
            predict_vocalization=True,
            predict_energy=True,
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.unfreeze_encoder()

        groups = model.get_parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

        encoder_groups = [g for g in groups if g["lr"] == 1e-5]
        assert len(encoder_groups) == 1, "Unfrozen encoder should have encoder group"

        # Can pass to optimizer without error
        optimizer = torch.optim.Adam(groups)
        assert optimizer is not None

    def test_wavlm_classifier_predict_event(self):
        """predict_event should return a valid (event_class, confidence) tuple."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.eval()

        audio = torch.randn(1, 16000)
        event_class, confidence = model.predict_event(audio)

        assert isinstance(event_class, str)
        assert event_class in config.event_classes
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_wavlm_classifier_event_probs_sum_to_one(self):
        """Softmax event probabilities should sum to ~1 along class dim."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.eval()

        batch = 3
        audio = torch.randn(batch, 16000)

        with torch.no_grad():
            out = model(audio)

        sums = out["event_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch), atol=1e-5)

    def test_wavlm_classifier_without_optional_heads(self):
        """When optional heads are disabled, those keys should not appear."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
            predict_vocalization=False,
            predict_energy=False,
        )
        model = _build_classifier_with_fake_wavlm(config)
        model.eval()

        batch = 2
        audio = torch.randn(batch, 16000)

        with torch.no_grad():
            out = model(audio)

        expected_keys = {"event_logits", "event_probs", "features", "hidden_features"}
        assert set(out.keys()) == expected_keys
        assert model.vocalization_head is None
        assert model.energy_head is None


# ============================================================================
# Backward compatibility tests
# ============================================================================

class TestBackwardCompatibility:
    """Tests ensuring Whisper encoder still works alongside the new WavLM encoder."""

    @pytest.fixture(autouse=True)
    def _patch_whisper_cache_dir(self, tmp_path):
        """Prevent tests from touching the real Whisper cache directory."""
        with patch(
            "synchronai.models.audio.whisper_encoder.get_whisper_cache_dir",
            return_value=tmp_path / "whisper_cache",
        ):
            yield

    def test_whisper_config_still_works(self):
        """AudioClassifierConfig() with defaults should still use Whisper."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        cfg = AudioClassifierConfig()
        assert cfg.encoder_type == "whisper"
        assert cfg.encoder_dim == 1280  # large-v3

    def test_explicit_whisper_encoder_type(self):
        """encoder_type='whisper' should create a WhisperEncoderFeatures."""
        from synchronai.models.audio.audio_classifier import AudioClassifier, AudioClassifierConfig
        from synchronai.models.audio.whisper_encoder import WhisperEncoderFeatures

        config = AudioClassifierConfig(encoder_type="whisper", whisper_model_size="tiny")

        with patch(
            "synchronai.models.audio.whisper_encoder.WhisperEncoderFeatures._load_model",
        ):
            model = AudioClassifier(config)

        assert isinstance(model.encoder, WhisperEncoderFeatures)

    def test_wavlm_encoder_type_creates_wavlm(self):
        """encoder_type='wavlm' should create a WavLMEncoderFeatures."""
        from synchronai.models.audio.audio_classifier import AudioClassifier, AudioClassifierConfig
        from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures

        config = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
        )

        with patch(
            "synchronai.models.audio.wavlm_encoder.WavLMEncoderFeatures._load_model",
        ):
            model = AudioClassifier(config)

        assert isinstance(model.encoder, WavLMEncoderFeatures)

    def test_wavlm_config_encoder_dim(self):
        """AudioClassifierConfig with wavlm encoder_type should report WavLM dim."""
        from synchronai.models.audio.audio_classifier import AudioClassifierConfig

        cfg = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-large",
        )
        assert cfg.encoder_dim == 1024

        cfg_base = AudioClassifierConfig(
            encoder_type="wavlm",
            wavlm_model_name="microsoft/wavlm-base-plus",
        )
        assert cfg_base.encoder_dim == 768


# ============================================================================
# WAVLM_CONFIGS constant tests
# ============================================================================

class TestWavLMDimConstants:
    """Tests for the WAVLM_CONFIGS mapping."""

    def test_all_expected_models_present(self):
        from synchronai.models.audio.wavlm_encoder import WAVLM_CONFIGS

        expected_keys = {
            "microsoft/wavlm-large",
            "microsoft/wavlm-base-plus",
            "microsoft/wavlm-base",
            "wavlm-large",
            "wavlm-base-plus",
            "wavlm-base",
        }
        assert expected_keys == set(WAVLM_CONFIGS.keys())

    def test_dims_are_positive_integers(self):
        from synchronai.models.audio.wavlm_encoder import WAVLM_CONFIGS

        for name, config in WAVLM_CONFIGS.items():
            assert isinstance(config["hidden_size"], int), (
                f"hidden_size for {name} should be int"
            )
            assert config["hidden_size"] > 0, (
                f"hidden_size for {name} should be positive"
            )
            assert isinstance(config["num_layers"], int), (
                f"num_layers for {name} should be int"
            )
            assert config["num_layers"] > 0, (
                f"num_layers for {name} should be positive"
            )

    def test_short_name_resolution(self):
        """Short names should resolve to the same full HuggingFace model IDs."""
        from synchronai.models.audio.wavlm_encoder import WAVLM_MODEL_MAP, _resolve_model_name

        assert _resolve_model_name("wavlm-large") == "microsoft/wavlm-large"
        assert _resolve_model_name("wavlm-base-plus") == "microsoft/wavlm-base-plus"
        assert _resolve_model_name("wavlm-base") == "microsoft/wavlm-base"

        # Full names should pass through unchanged
        assert _resolve_model_name("microsoft/wavlm-large") == "microsoft/wavlm-large"

    def test_short_and_full_names_have_same_configs(self):
        """Short alias configs should match their full HuggingFace name configs."""
        from synchronai.models.audio.wavlm_encoder import WAVLM_CONFIGS

        assert WAVLM_CONFIGS["wavlm-large"] == WAVLM_CONFIGS["microsoft/wavlm-large"]
        assert WAVLM_CONFIGS["wavlm-base-plus"] == WAVLM_CONFIGS["microsoft/wavlm-base-plus"]
        assert WAVLM_CONFIGS["wavlm-base"] == WAVLM_CONFIGS["microsoft/wavlm-base"]


# ============================================================================
# get_wavlm_encoder factory and caching tests
# ============================================================================

class TestGetWavLMEncoder:
    """Tests for get_wavlm_encoder factory and caching."""

    def test_returns_wavlm_encoder_features(self):
        from synchronai.models.audio.wavlm_encoder import (
            WavLMEncoderFeatures,
            get_wavlm_encoder,
        )

        encoder = get_wavlm_encoder(model_name="wavlm-large", use_cache=False)
        assert isinstance(encoder, WavLMEncoderFeatures)

    def test_caching_returns_same_instance(self):
        from synchronai.models.audio.wavlm_encoder import get_wavlm_encoder

        enc1 = get_wavlm_encoder(model_name="wavlm-large", use_cache=True)
        enc2 = get_wavlm_encoder(model_name="wavlm-large", use_cache=True)
        assert enc1 is enc2

    def test_different_config_creates_new_instance(self):
        from synchronai.models.audio.wavlm_encoder import get_wavlm_encoder

        enc1 = get_wavlm_encoder(model_name="wavlm-large", freeze=True, use_cache=True)
        enc2 = get_wavlm_encoder(model_name="wavlm-large", freeze=False, use_cache=True)
        assert enc1 is not enc2

    def test_different_model_creates_new_instance(self):
        from synchronai.models.audio.wavlm_encoder import get_wavlm_encoder

        enc1 = get_wavlm_encoder(model_name="wavlm-large", use_cache=True)
        enc2 = get_wavlm_encoder(model_name="wavlm-base-plus", use_cache=True)
        assert enc1 is not enc2

    def test_clear_cache_resets(self):
        from synchronai.models.audio.wavlm_encoder import (
            get_wavlm_encoder,
            clear_wavlm_cache,
        )

        enc1 = get_wavlm_encoder(model_name="wavlm-large", use_cache=True)
        clear_wavlm_cache()
        enc2 = get_wavlm_encoder(model_name="wavlm-large", use_cache=True)
        assert enc1 is not enc2

    def test_no_cache_always_creates_new(self):
        from synchronai.models.audio.wavlm_encoder import get_wavlm_encoder

        enc1 = get_wavlm_encoder(model_name="wavlm-large", use_cache=False)
        enc2 = get_wavlm_encoder(model_name="wavlm-large", use_cache=False)
        assert enc1 is not enc2

    def test_short_name_resolves_in_factory(self):
        """Short names passed to get_wavlm_encoder should resolve correctly."""
        from synchronai.models.audio.wavlm_encoder import get_wavlm_encoder

        encoder = get_wavlm_encoder(model_name="wavlm-base-plus", use_cache=False)
        assert encoder.model_name == "microsoft/wavlm-base-plus"
        assert encoder.encoder_dim == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

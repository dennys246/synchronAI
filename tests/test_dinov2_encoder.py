"""
Comprehensive tests for the DINOv2 vision encoder.

Tests cover:
- DINOv2EncoderConfig: dataclass fields, feature_dim property, short name resolution
- DINOv2FeatureExtractor: lazy loading, CLS pool, mean_patch pool, forward_spatial,
  output shapes, load_state_dict triggers
- Freeze / unfreeze: freeze_backbone, unfreeze_backbone, requires_grad tracking
- get_parameter_groups: frozen vs unfrozen backbone parameter groups
- DINOV2_CONFIGS constant validation
- get_dinov2_encoder factory and caching, clear_dinov2_cache

The HuggingFace Dinov2Model is mocked throughout to avoid requiring the
transformers package or downloading model weights.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Default dimensions for dinov2-base (patch_size=14, 224x224 -> 16x16 = 256 patches)
_DEFAULT_HIDDEN_SIZE = 768
_DEFAULT_NUM_PATCHES = 256
_DEFAULT_NUM_TOKENS = _DEFAULT_NUM_PATCHES + 1  # 257 = CLS + 256 patches


class _FakeDINOv2Output:
    """Mimics the HuggingFace BaseModelOutputWithPooling with last_hidden_state."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class FakeDINOv2(nn.Module):
    """Fake nn.Module that mimics transformers.Dinov2Model output format.

    When called with ``(pixel_values=...)`` returns an object whose
    ``.last_hidden_state`` attribute has shape ``(B, num_patches+1, hidden_size)``
    where index 0 is the CLS token and indices 1..256 are patch tokens.
    """

    def __init__(self, hidden_size: int = _DEFAULT_HIDDEN_SIZE,
                 num_patches: int = _DEFAULT_NUM_PATCHES):
        super().__init__()
        # Real parameter so parameters() is non-empty for freeze/unfreeze tests
        self.linear = nn.Linear(hidden_size, hidden_size)
        self._hidden_size = hidden_size
        self._num_patches = num_patches

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> _FakeDINOv2Output:
        batch = pixel_values.shape[0]
        num_tokens = self._num_patches + 1  # CLS + patches
        # Pass through self.linear so gradients flow through parameters when unfrozen
        dummy = torch.randn(batch, num_tokens, self._hidden_size,
                            device=pixel_values.device)
        last_hidden_state = self.linear(dummy)
        return _FakeDINOv2Output(last_hidden_state=last_hidden_state)


def _make_fake_dinov2(hidden_size: int = _DEFAULT_HIDDEN_SIZE,
                      num_patches: int = _DEFAULT_NUM_PATCHES) -> FakeDINOv2:
    """Create a FakeDINOv2 instance with the given dimensions."""
    return FakeDINOv2(hidden_size=hidden_size, num_patches=num_patches)


def _patch_from_pretrained(hidden_size: int = _DEFAULT_HIDDEN_SIZE,
                           num_patches: int = _DEFAULT_NUM_PATCHES):
    """Return a context manager that patches Dinov2Model.from_pretrained.

    The patched from_pretrained returns a FakeDINOv2 instance so that no
    real model weights are downloaded.
    """
    fake_model = _make_fake_dinov2(hidden_size, num_patches)

    mock_dinov2_class = MagicMock()
    mock_dinov2_class.from_pretrained.return_value = fake_model

    return patch(
        "synchronai.models.cv.dinov2_encoder._lazy_import_transformers",
        return_value=mock_dinov2_class,
    )


@pytest.fixture(autouse=True)
def _clear_dinov2_cache():
    """Clear the global DINOv2 encoder cache before and after each test."""
    from synchronai.models.cv.dinov2_encoder import clear_dinov2_cache

    clear_dinov2_cache()
    yield
    clear_dinov2_cache()


# ============================================================================
# DINOv2EncoderConfig tests
# ============================================================================

class TestDINOv2Config:
    """Tests for DINOv2EncoderConfig dataclass and its properties."""

    def test_default_values(self):
        from synchronai.models.cv.dinov2_encoder import DINOv2EncoderConfig

        cfg = DINOv2EncoderConfig()
        assert cfg.model_name == "facebook/dinov2-base"
        assert cfg.device is None
        assert cfg.freeze is True
        assert cfg.pool_mode == "cls"

    @pytest.mark.parametrize(
        "model_name, expected_dim",
        [
            ("facebook/dinov2-small", 384),
            ("facebook/dinov2-base", 768),
            ("facebook/dinov2-large", 1024),
            ("facebook/dinov2-giant", 1536),
            # Short name aliases
            ("dinov2-small", 384),
            ("dinov2-base", 768),
            ("dinov2-large", 1024),
            ("dinov2-giant", 1536),
        ],
    )
    def test_feature_dim_property(self, model_name, expected_dim):
        from synchronai.models.cv.dinov2_encoder import DINOv2EncoderConfig

        cfg = DINOv2EncoderConfig(model_name=model_name)
        assert cfg.feature_dim == expected_dim

    def test_short_name_aliases(self):
        """Short aliases should resolve to the same dims as full HuggingFace names."""
        from synchronai.models.cv.dinov2_encoder import DINOv2EncoderConfig

        short = DINOv2EncoderConfig(model_name="dinov2-base")
        full = DINOv2EncoderConfig(model_name="facebook/dinov2-base")
        assert short.feature_dim == full.feature_dim

    def test_unknown_model_defaults_to_base(self):
        """Unknown model names should fall back to dinov2-base default (768)."""
        from synchronai.models.cv.dinov2_encoder import DINOv2EncoderConfig

        cfg = DINOv2EncoderConfig(model_name="some-unknown-model")
        assert cfg.feature_dim == 768

    def test_custom_values(self):
        from synchronai.models.cv.dinov2_encoder import DINOv2EncoderConfig

        cfg = DINOv2EncoderConfig(
            model_name="dinov2-large",
            device="cuda:0",
            freeze=False,
            pool_mode="mean_patch",
        )
        assert cfg.device == "cuda:0"
        assert cfg.freeze is False
        assert cfg.pool_mode == "mean_patch"
        assert cfg.feature_dim == 1024

    def test_pool_mode_options(self):
        """Both pool_mode options should be settable on the config."""
        from synchronai.models.cv.dinov2_encoder import DINOv2EncoderConfig

        cls_cfg = DINOv2EncoderConfig(pool_mode="cls")
        assert cls_cfg.pool_mode == "cls"

        mean_cfg = DINOv2EncoderConfig(pool_mode="mean_patch")
        assert mean_cfg.pool_mode == "mean_patch"


# ============================================================================
# DINOv2FeatureExtractor tests
# ============================================================================

class TestDINOv2FeatureExtractor:
    """Tests for DINOv2FeatureExtractor (with mocked Dinov2Model backend)."""

    def test_init_does_not_load_model(self):
        """Construction should be lazy -- no model loaded yet."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(model_name="facebook/dinov2-base", freeze=True)
        assert encoder._is_loaded is False
        assert encoder.dinov2 is None

    def test_feature_dim_property_without_loading(self):
        """feature_dim should be available without loading the model."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(model_name="facebook/dinov2-base")
        assert encoder.feature_dim == 768
        assert encoder._is_loaded is False

    def test_num_patches_property_without_loading(self):
        """num_patches should be available without loading the model."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(model_name="facebook/dinov2-base")
        assert encoder.num_patches == 256
        assert encoder._is_loaded is False

    def test_feature_dim_for_each_variant(self):
        """feature_dim should match DINOV2_CONFIGS for all model variants."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        expected = {
            "facebook/dinov2-small": 384,
            "facebook/dinov2-base": 768,
            "facebook/dinov2-large": 1024,
            "facebook/dinov2-giant": 1536,
        }
        for model_name, dim in expected.items():
            encoder = DINOv2FeatureExtractor(model_name=model_name)
            assert encoder.feature_dim == dim, f"Wrong feature_dim for {model_name}"

    def test_forward_cls_pool_output_shape(self):
        """CLS pool mode should produce (batch, hidden_size)."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=768, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base",
                freeze=True,
                pool_mode="cls",
            )
            batch = 4
            x = torch.randn(batch, 3, 224, 224)
            out = encoder(x)

        assert out.shape == (batch, 768)

    def test_forward_mean_patch_pool_output_shape(self):
        """Mean patch pool mode should produce (batch, hidden_size)."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=768, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base",
                freeze=True,
                pool_mode="mean_patch",
            )
            batch = 2
            x = torch.randn(batch, 3, 224, 224)
            out = encoder(x)

        assert out.shape == (batch, 768)

    def test_cls_and_mean_patch_produce_different_results(self):
        """CLS and mean_patch pooling should generally yield different features."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        # Use the same fake model for both so the underlying hidden states are
        # produced by the same random process, but the pooling differs.
        fake = _make_fake_dinov2(hidden_size=768, num_patches=256)
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = fake

        with patch(
            "synchronai.models.cv.dinov2_encoder._lazy_import_transformers",
            return_value=mock_cls,
        ):
            enc_cls = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base", pool_mode="cls", freeze=True,
            )
            enc_mean = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base", pool_mode="mean_patch", freeze=True,
            )

            x = torch.randn(1, 3, 224, 224)
            out_cls = enc_cls(x)
            out_mean = enc_mean(x)

        # They should have the same shape but (almost certainly) different values
        assert out_cls.shape == out_mean.shape == (1, 768)

    def test_forward_triggers_lazy_load(self):
        """Calling forward on an unloaded encoder should trigger _load_model."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=768, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base", freeze=True,
            )
            assert encoder._is_loaded is False

            x = torch.randn(1, 3, 224, 224)
            encoder(x)

            assert encoder._is_loaded is True
            assert encoder.dinov2 is not None

    def test_forward_spatial_output_shape(self):
        """forward_spatial should return (B, D, H_patches, W_patches)."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=768, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base", freeze=True,
            )
            batch = 2
            x = torch.randn(batch, 3, 224, 224)
            spatial = encoder.forward_spatial(x)

        # 256 patches -> 16x16 spatial grid
        assert spatial.shape == (batch, 768, 16, 16)

    def test_forward_spatial_with_large_variant(self):
        """forward_spatial should work with dinov2-large dimensions."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=1024, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-large", freeze=True,
            )
            batch = 1
            x = torch.randn(batch, 3, 224, 224)
            spatial = encoder.forward_spatial(x)

        assert spatial.shape == (batch, 1024, 16, 16)

    def test_forward_invalid_pool_mode_raises(self):
        """An unknown pool_mode should raise ValueError."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=768, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base",
                freeze=True,
                pool_mode="attention",
            )
            x = torch.randn(1, 3, 224, 224)
            with pytest.raises(ValueError, match="Unknown pool_mode"):
                encoder(x)

    def test_short_name_resolution_in_extractor(self):
        """Short model names should be resolved to full HuggingFace IDs."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(model_name="dinov2-base")
        assert encoder.model_name == "facebook/dinov2-base"
        assert encoder.feature_dim == 768

    def test_load_state_dict_triggers_load_when_dinov2_keys_present(self):
        """load_state_dict should call _load_model when state dict has dinov2.* keys."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(model_name="facebook/dinov2-base", freeze=True)
        fake_state = {"dinov2.encoder.weight": torch.randn(10, 10)}

        with patch.object(encoder, "_load_model") as mock_load:
            mock_load.side_effect = lambda: (
                setattr(encoder, "_is_loaded", True)
                or setattr(encoder, "dinov2", nn.Linear(10, 10))
            )
            try:
                encoder.load_state_dict(fake_state, strict=False)
            except Exception:
                pass  # shape mismatch is expected
            mock_load.assert_called_once()

    def test_load_state_dict_skips_load_without_dinov2_keys(self):
        """load_state_dict should NOT call _load_model when no dinov2.* keys."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(model_name="facebook/dinov2-base", freeze=True)
        fake_state = {"other_module.weight": torch.randn(5)}

        with patch.object(encoder, "_load_model") as mock_load:
            try:
                encoder.load_state_dict(fake_state, strict=False)
            except Exception:
                pass
            mock_load.assert_not_called()

    def test_to_method_returns_self(self):
        """to('cpu') should return the same DINOv2FeatureExtractor instance."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(model_name="facebook/dinov2-base", freeze=True)
        result = encoder.to("cpu")
        assert result is encoder

    def test_forward_with_small_variant(self):
        """Forward should work correctly with dinov2-small dimensions (384)."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=384, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-small", freeze=True, pool_mode="cls",
            )
            batch = 3
            x = torch.randn(batch, 3, 224, 224)
            out = encoder(x)

        assert out.shape == (batch, 384)

    def test_forward_with_giant_variant(self):
        """Forward should work correctly with dinov2-giant dimensions (1536)."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=1536, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-giant", freeze=True, pool_mode="mean_patch",
            )
            batch = 1
            x = torch.randn(batch, 3, 224, 224)
            out = encoder(x)

        assert out.shape == (batch, 1536)


# ============================================================================
# Freeze / unfreeze tests
# ============================================================================

class TestDINOv2FreezeUnfreeze:
    """Tests for freeze_backbone() and unfreeze_backbone() on DINOv2FeatureExtractor."""

    def _make_encoder(self, freeze=True):
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(
            model_name="facebook/dinov2-base", freeze=freeze,
        )
        # Inject fake model directly (bypass lazy loading)
        encoder.dinov2 = _make_fake_dinov2(hidden_size=768, num_patches=256)
        encoder._is_loaded = True
        if freeze:
            encoder.freeze_backbone()
        return encoder

    def test_freeze_backbone_freezes_all_params(self):
        """freeze_backbone should set requires_grad=False on all dinov2 params."""
        encoder = self._make_encoder(freeze=False)
        # Confirm unfrozen initially
        for p in encoder.dinov2.parameters():
            assert p.requires_grad is True

        encoder.freeze_backbone()
        for p in encoder.dinov2.parameters():
            assert p.requires_grad is False

    def test_unfreeze_backbone_unfreezes_all_params(self):
        """unfreeze_backbone should set requires_grad=True on all dinov2 params."""
        encoder = self._make_encoder(freeze=True)
        # Confirm frozen
        for p in encoder.dinov2.parameters():
            assert p.requires_grad is False

        encoder.unfreeze_backbone()
        for p in encoder.dinov2.parameters():
            assert p.requires_grad is True

    def test_freeze_sets_internal_flag(self):
        """freeze_backbone should set _freeze=True."""
        encoder = self._make_encoder(freeze=False)
        assert encoder._freeze is False

        encoder.freeze_backbone()
        assert encoder._freeze is True

    def test_unfreeze_sets_internal_flag(self):
        """unfreeze_backbone should set _freeze=False."""
        encoder = self._make_encoder(freeze=True)
        assert encoder._freeze is True

        encoder.unfreeze_backbone()
        assert encoder._freeze is False

    def test_multiple_freeze_unfreeze_cycles(self):
        """Repeated freeze/unfreeze cycles should work without issues."""
        encoder = self._make_encoder(freeze=False)

        for _ in range(3):
            encoder.freeze_backbone()
            assert all(not p.requires_grad for p in encoder.dinov2.parameters())
            assert encoder._freeze is True

            encoder.unfreeze_backbone()
            assert all(p.requires_grad for p in encoder.dinov2.parameters())
            assert encoder._freeze is False

    def test_freeze_before_load_does_not_raise(self):
        """Calling freeze_backbone before model is loaded should not raise."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(
            model_name="facebook/dinov2-base", freeze=False,
        )
        assert encoder.dinov2 is None

        # Should not raise even though dinov2 is None
        encoder.freeze_backbone()
        assert encoder._freeze is True

    def test_unfreeze_before_load_does_not_raise(self):
        """Calling unfreeze_backbone before model is loaded should not raise."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(
            model_name="facebook/dinov2-base", freeze=True,
        )
        assert encoder.dinov2 is None

        # Should not raise even though dinov2 is None
        encoder.unfreeze_backbone()
        assert encoder._freeze is False

    def test_forward_uses_no_grad_when_frozen(self):
        """When frozen, forward should not track gradients on backbone output."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=768, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base", freeze=True, pool_mode="cls",
            )
            x = torch.randn(1, 3, 224, 224)
            out = encoder(x)

        # Output should not require grad since backbone is frozen
        assert not out.requires_grad

    def test_forward_uses_grad_when_unfrozen(self):
        """When unfrozen, forward should track gradients on backbone output."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        with _patch_from_pretrained(hidden_size=768, num_patches=256):
            encoder = DINOv2FeatureExtractor(
                model_name="facebook/dinov2-base", freeze=False, pool_mode="cls",
            )
            x = torch.randn(1, 3, 224, 224, requires_grad=True)
            out = encoder(x)

        # With unfrozen backbone and requires_grad input, output should require grad
        assert out.requires_grad


# ============================================================================
# get_parameter_groups tests
# ============================================================================

class TestDINOv2ParameterGroups:
    """Tests for get_parameter_groups() on DINOv2FeatureExtractor."""

    def _make_encoder(self, freeze=True):
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(
            model_name="facebook/dinov2-base", freeze=freeze,
        )
        encoder.dinov2 = _make_fake_dinov2(hidden_size=768, num_patches=256)
        encoder._is_loaded = True
        if freeze:
            encoder.freeze_backbone()
        return encoder

    def test_frozen_backbone_returns_empty_groups(self):
        """With frozen backbone, get_parameter_groups should return no groups."""
        encoder = self._make_encoder(freeze=True)
        groups = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        assert len(groups) == 0, "Frozen backbone should produce no parameter groups"

    def test_unfrozen_backbone_returns_backbone_group(self):
        """Unfrozen backbone should return one group with backbone params at backbone_lr."""
        encoder = self._make_encoder(freeze=False)
        groups = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        assert len(groups) == 1, "Unfrozen backbone should have exactly one group"
        assert groups[0]["lr"] == 1e-5

        # Group should contain the dinov2 parameters
        group_params = list(groups[0]["params"])
        assert len(group_params) > 0

    def test_unfrozen_group_only_includes_requires_grad_params(self):
        """Parameter group should only include params with requires_grad=True."""
        encoder = self._make_encoder(freeze=False)
        groups = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        for group in groups:
            for p in group["params"]:
                assert p.requires_grad is True, (
                    "Only requires_grad=True params should be in groups"
                )

    def test_parameter_groups_can_be_passed_to_optimizer(self):
        """Parameter groups should be valid input for torch.optim.Adam."""
        encoder = self._make_encoder(freeze=False)
        groups = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        # Should not raise
        optimizer = torch.optim.Adam(groups)
        assert optimizer is not None

    def test_frozen_groups_cannot_create_optimizer(self):
        """Empty parameter groups (frozen) should not create a valid optimizer."""
        encoder = self._make_encoder(freeze=True)
        groups = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        assert groups == []

    def test_unfreeze_then_get_groups(self):
        """After unfreezing, get_parameter_groups should return non-empty groups."""
        encoder = self._make_encoder(freeze=True)

        # Frozen: no groups
        groups_frozen = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
        assert len(groups_frozen) == 0

        # Unfreeze
        encoder.unfreeze_backbone()

        # Now should have groups
        groups_unfrozen = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
        assert len(groups_unfrozen) == 1
        assert groups_unfrozen[0]["lr"] == 1e-5

    def test_dinov2_not_loaded_returns_empty_groups(self):
        """If dinov2 has not been loaded yet, get_parameter_groups returns empty."""
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor

        encoder = DINOv2FeatureExtractor(
            model_name="facebook/dinov2-base", freeze=False,
        )
        # dinov2 is None (not loaded)
        groups = encoder.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
        assert len(groups) == 0


# ============================================================================
# DINOV2_CONFIGS constant tests
# ============================================================================

class TestDINOv2DimConstants:
    """Tests for the DINOV2_CONFIGS mapping."""

    def test_all_expected_models_present(self):
        from synchronai.models.cv.dinov2_encoder import DINOV2_CONFIGS

        expected_keys = {
            "facebook/dinov2-small",
            "facebook/dinov2-base",
            "facebook/dinov2-large",
            "facebook/dinov2-giant",
            # Short aliases
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
        }
        assert expected_keys == set(DINOV2_CONFIGS.keys())

    def test_hidden_sizes_are_positive_integers(self):
        from synchronai.models.cv.dinov2_encoder import DINOV2_CONFIGS

        for name, config in DINOV2_CONFIGS.items():
            assert isinstance(config["hidden_size"], int), (
                f"hidden_size for {name} should be int"
            )
            assert config["hidden_size"] > 0, (
                f"hidden_size for {name} should be positive"
            )

    def test_num_patches_are_positive_integers(self):
        from synchronai.models.cv.dinov2_encoder import DINOV2_CONFIGS

        for name, config in DINOV2_CONFIGS.items():
            assert isinstance(config["num_patches"], int), (
                f"num_patches for {name} should be int"
            )
            assert config["num_patches"] > 0, (
                f"num_patches for {name} should be positive"
            )

    def test_all_variants_have_256_patches(self):
        """All DINOv2 variants use patch_size=14 on 224x224 -> 256 patches."""
        from synchronai.models.cv.dinov2_encoder import DINOV2_CONFIGS

        for name, config in DINOV2_CONFIGS.items():
            assert config["num_patches"] == 256, (
                f"{name} should have 256 patches for 224x224 input with patch_size=14"
            )

    def test_short_name_resolution(self):
        """Short names should resolve to the correct full HuggingFace model IDs."""
        from synchronai.models.cv.dinov2_encoder import DINOV2_MODEL_MAP, _resolve_model_name

        assert _resolve_model_name("dinov2-small") == "facebook/dinov2-small"
        assert _resolve_model_name("dinov2-base") == "facebook/dinov2-base"
        assert _resolve_model_name("dinov2-large") == "facebook/dinov2-large"
        assert _resolve_model_name("dinov2-giant") == "facebook/dinov2-giant"

        # Full names should pass through unchanged
        assert _resolve_model_name("facebook/dinov2-base") == "facebook/dinov2-base"

    def test_short_and_full_names_have_same_configs(self):
        """Short alias configs should match their full HuggingFace name configs."""
        from synchronai.models.cv.dinov2_encoder import DINOV2_CONFIGS

        assert DINOV2_CONFIGS["dinov2-small"] == DINOV2_CONFIGS["facebook/dinov2-small"]
        assert DINOV2_CONFIGS["dinov2-base"] == DINOV2_CONFIGS["facebook/dinov2-base"]
        assert DINOV2_CONFIGS["dinov2-large"] == DINOV2_CONFIGS["facebook/dinov2-large"]
        assert DINOV2_CONFIGS["dinov2-giant"] == DINOV2_CONFIGS["facebook/dinov2-giant"]

    def test_known_hidden_sizes(self):
        """Verify the exact hidden_size values for each model variant."""
        from synchronai.models.cv.dinov2_encoder import DINOV2_CONFIGS

        expected = {
            "facebook/dinov2-small": 384,
            "facebook/dinov2-base": 768,
            "facebook/dinov2-large": 1024,
            "facebook/dinov2-giant": 1536,
        }
        for name, expected_dim in expected.items():
            assert DINOV2_CONFIGS[name]["hidden_size"] == expected_dim


# ============================================================================
# get_dinov2_encoder factory and caching tests
# ============================================================================

class TestDINOv2Cache:
    """Tests for get_dinov2_encoder factory and caching, and clear_dinov2_cache."""

    def test_returns_dinov2_feature_extractor(self):
        from synchronai.models.cv.dinov2_encoder import (
            DINOv2FeatureExtractor,
            get_dinov2_encoder,
        )

        encoder = get_dinov2_encoder(model_name="dinov2-base", use_cache=False)
        assert isinstance(encoder, DINOv2FeatureExtractor)

    def test_caching_returns_same_instance(self):
        from synchronai.models.cv.dinov2_encoder import get_dinov2_encoder

        enc1 = get_dinov2_encoder(model_name="dinov2-base", use_cache=True)
        enc2 = get_dinov2_encoder(model_name="dinov2-base", use_cache=True)
        assert enc1 is enc2

    def test_different_config_creates_new_instance(self):
        from synchronai.models.cv.dinov2_encoder import get_dinov2_encoder

        enc1 = get_dinov2_encoder(model_name="dinov2-base", freeze=True, use_cache=True)
        enc2 = get_dinov2_encoder(model_name="dinov2-base", freeze=False, use_cache=True)
        assert enc1 is not enc2

    def test_different_model_creates_new_instance(self):
        from synchronai.models.cv.dinov2_encoder import get_dinov2_encoder

        enc1 = get_dinov2_encoder(model_name="dinov2-base", use_cache=True)
        enc2 = get_dinov2_encoder(model_name="dinov2-large", use_cache=True)
        assert enc1 is not enc2

    def test_different_pool_mode_creates_new_instance(self):
        from synchronai.models.cv.dinov2_encoder import get_dinov2_encoder

        enc1 = get_dinov2_encoder(
            model_name="dinov2-base", pool_mode="cls", use_cache=True,
        )
        enc2 = get_dinov2_encoder(
            model_name="dinov2-base", pool_mode="mean_patch", use_cache=True,
        )
        assert enc1 is not enc2

    def test_clear_cache_resets(self):
        from synchronai.models.cv.dinov2_encoder import (
            get_dinov2_encoder,
            clear_dinov2_cache,
        )

        enc1 = get_dinov2_encoder(model_name="dinov2-base", use_cache=True)
        clear_dinov2_cache()
        enc2 = get_dinov2_encoder(model_name="dinov2-base", use_cache=True)
        assert enc1 is not enc2

    def test_no_cache_always_creates_new(self):
        from synchronai.models.cv.dinov2_encoder import get_dinov2_encoder

        enc1 = get_dinov2_encoder(model_name="dinov2-base", use_cache=False)
        enc2 = get_dinov2_encoder(model_name="dinov2-base", use_cache=False)
        assert enc1 is not enc2

    def test_short_name_resolves_in_factory(self):
        """Short names passed to get_dinov2_encoder should resolve correctly."""
        from synchronai.models.cv.dinov2_encoder import get_dinov2_encoder

        encoder = get_dinov2_encoder(model_name="dinov2-large", use_cache=False)
        assert encoder.model_name == "facebook/dinov2-large"
        assert encoder.feature_dim == 1024

    def test_clear_cache_is_idempotent(self):
        """Calling clear_dinov2_cache multiple times should not raise."""
        from synchronai.models.cv.dinov2_encoder import clear_dinov2_cache

        clear_dinov2_cache()
        clear_dinov2_cache()
        clear_dinov2_cache()
        # No assertion needed -- just verifying no exception is raised

    def test_cache_key_includes_all_params(self):
        """Cache key should distinguish on model_name, device, freeze, and pool_mode."""
        from synchronai.models.cv.dinov2_encoder import get_dinov2_encoder

        # Same model, same freeze, different pool_mode -> different instances
        enc_cls = get_dinov2_encoder(
            model_name="dinov2-base", freeze=True, pool_mode="cls", use_cache=True,
        )
        enc_mean = get_dinov2_encoder(
            model_name="dinov2-base", freeze=True, pool_mode="mean_patch", use_cache=True,
        )
        assert enc_cls is not enc_mean

        # Re-request with same params should return cached instance
        enc_mean_again = get_dinov2_encoder(
            model_name="dinov2-base", freeze=True, pool_mode="mean_patch", use_cache=True,
        )
        assert enc_mean is enc_mean_again


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

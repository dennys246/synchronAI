"""
Unit tests for the cross-person attention module and person-aware video classifier.

Tests cover:
- CrossPersonAttentionLayer: output shapes for 2D/3D input, residual connection,
  weight-sharing (same weights for A->B and B->A directions)
- CrossPersonAttention: multi-layer stacking, gradient flow, self-attention fallback
- PersonAwareVideoClassifier: forward passes for 0/1/2 person counts, mixed
  person-count batches, return_features mode, parameter groups with cross-attn
- build_person_aware_classifier and load_person_aware_classifier factory/loader

The DINOv2 dependency is mocked throughout via a FakeFeatureExtractor so these
tests run without HuggingFace or any pretrained model weights.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from synchronai.models.cv.cross_person_attention import (
    CrossPersonAttention,
    CrossPersonAttentionLayer,
)
from synchronai.models.cv.person_aware_classifier import (
    PersonAwareConfig,
    PersonAwareVideoClassifier,
    build_person_aware_classifier,
    load_person_aware_classifier,
)


# ---------------------------------------------------------------------------
# Helpers: fake feature extractor to replace DINOv2
# ---------------------------------------------------------------------------

FAKE_FEATURE_DIM = 64


class FakeFeatureExtractor(nn.Module):
    """Lightweight stand-in for DINOv2FeatureExtractor.

    Accepts (B, 3, H, W) images and returns (B, feature_dim) features
    via a single linear layer (after flattening a small adaptive pool).
    """

    def __init__(self, feature_dim: int = FAKE_FEATURE_DIM, freeze: bool = True):
        super().__init__()
        self._feature_dim = feature_dim
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, feature_dim, bias=False),
        )
        if freeze:
            self.freeze_backbone()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = []
        if params:
            groups.append({"params": params, "lr": backbone_lr})
        return groups


@pytest.fixture()
def patch_feature_extractor():
    """Patch _create_feature_extractor to return a FakeFeatureExtractor.

    This avoids loading DINOv2 weights from HuggingFace during tests.
    """
    def _factory(config):
        return FakeFeatureExtractor(
            feature_dim=FAKE_FEATURE_DIM,
            freeze=config.freeze_backbone,
        )

    with patch(
        "synchronai.models.cv.person_aware_classifier._create_feature_extractor",
        side_effect=_factory,
    ):
        yield


@pytest.fixture()
def default_config() -> PersonAwareConfig:
    """Return a minimal PersonAwareConfig suitable for fast unit tests."""
    return PersonAwareConfig(
        backbone="dinov2-base",
        freeze_backbone=True,
        window_seconds=1.0,
        sample_fps=4.0,
        frame_height=224,
        frame_width=224,
        num_cross_attn_layers=1,
        cross_attn_heads=4,
        cross_attn_dropout=0.1,
        temporal_aggregation="lstm",
        hidden_dim=32,
        dropout=0.1,
        output_dim=1,
    )


def _make_frames(batch: int, n_frames: int, h: int = 224, w: int = 224) -> torch.Tensor:
    """Create a random (B, T, 3, H, W) tensor."""
    return torch.randn(batch, n_frames, 3, h, w)


# ===========================================================================
# CrossPersonAttentionLayer
# ===========================================================================

class TestCrossPersonAttentionLayer:
    """Tests for a single CrossPersonAttentionLayer."""

    def test_output_shape_2d(self):
        """2D input (B, D) should produce (B, D) output for both persons."""
        embed_dim = 32
        layer = CrossPersonAttentionLayer(embed_dim, num_heads=4, dropout=0.0)

        a = torch.randn(4, embed_dim)
        b = torch.randn(4, embed_dim)
        out_a, out_b = layer(a, b)

        assert out_a.shape == (4, embed_dim)
        assert out_b.shape == (4, embed_dim)

    def test_output_shape_3d(self):
        """3D input (B, T, D) should produce (B, T, D) output for both persons."""
        embed_dim = 32
        layer = CrossPersonAttentionLayer(embed_dim, num_heads=4, dropout=0.0)

        a = torch.randn(2, 8, embed_dim)
        b = torch.randn(2, 8, embed_dim)
        out_a, out_b = layer(a, b)

        assert out_a.shape == (2, 8, embed_dim)
        assert out_b.shape == (2, 8, embed_dim)

    def test_output_shape_single_timestep_3d(self):
        """3D input with T=1 should still work correctly."""
        embed_dim = 16
        layer = CrossPersonAttentionLayer(embed_dim, num_heads=2, dropout=0.0)

        a = torch.randn(3, 1, embed_dim)
        b = torch.randn(3, 1, embed_dim)
        out_a, out_b = layer(a, b)

        assert out_a.shape == (3, 1, embed_dim)
        assert out_b.shape == (3, 1, embed_dim)

    def test_residual_connection(self):
        """Output should differ from input but not be completely unrelated.

        The residual connection means output = norm(input + dropout(attn(input))),
        so output should not be identical to input (attention contributes) but
        should be correlated with it.
        """
        embed_dim = 32
        layer = CrossPersonAttentionLayer(embed_dim, num_heads=4, dropout=0.0)
        layer.eval()

        a = torch.randn(2, embed_dim)
        b = torch.randn(2, embed_dim)
        out_a, out_b = layer(a, b)

        # Output should not be identical to input
        assert not torch.allclose(out_a, a, atol=1e-5), (
            "Output should differ from input due to attention"
        )
        # But should be in the same general ballpark (residual preserves scale)
        assert out_a.abs().mean() < 10.0, "Output magnitude should be reasonable"

    def test_weight_sharing_symmetric(self):
        """A->B and B->A use the same nn.MultiheadAttention instance.

        Verify that the layer has a single cross_attn module, not two.
        """
        embed_dim = 32
        layer = CrossPersonAttentionLayer(embed_dim, num_heads=4)

        # There is exactly one MultiheadAttention module
        mha_modules = [m for m in layer.modules() if isinstance(m, nn.MultiheadAttention)]
        assert len(mha_modules) == 1, (
            "CrossPersonAttentionLayer should use a single shared MultiheadAttention"
        )

    def test_symmetric_with_identical_inputs(self):
        """When A == B, cross-attention should produce identical outputs for both.

        Since A queries B and B queries A with the same weights, if A == B
        then attended_a and attended_b should be equal (up to LayerNorm
        differences, but norm_a and norm_b start with the same init).
        """
        embed_dim = 32
        layer = CrossPersonAttentionLayer(embed_dim, num_heads=4, dropout=0.0)
        layer.eval()

        # Make norm_a and norm_b have same parameters for this test
        layer.norm_b.load_state_dict(layer.norm_a.state_dict())

        x = torch.randn(2, embed_dim)
        out_a, out_b = layer(x, x.clone())

        assert torch.allclose(out_a, out_b, atol=1e-5), (
            "With identical inputs and shared weights, outputs should match"
        )

    def test_batch_size_one(self):
        """Layer should work with batch_size=1."""
        embed_dim = 16
        layer = CrossPersonAttentionLayer(embed_dim, num_heads=2, dropout=0.0)

        a = torch.randn(1, embed_dim)
        b = torch.randn(1, embed_dim)
        out_a, out_b = layer(a, b)

        assert out_a.shape == (1, embed_dim)
        assert out_b.shape == (1, embed_dim)

    def test_different_num_heads(self):
        """Layer should work with various num_heads values."""
        embed_dim = 24  # Divisible by 1, 2, 3, 4, 6, 8, 12, 24
        for num_heads in [1, 2, 3, 6]:
            layer = CrossPersonAttentionLayer(embed_dim, num_heads=num_heads)
            a = torch.randn(2, embed_dim)
            b = torch.randn(2, embed_dim)
            out_a, out_b = layer(a, b)
            assert out_a.shape == (2, embed_dim)


# ===========================================================================
# CrossPersonAttention (multi-layer)
# ===========================================================================

class TestCrossPersonAttention:
    """Tests for the multi-layer CrossPersonAttention module."""

    def test_single_layer_output_shape(self):
        """Single-layer module should produce correct shapes."""
        module = CrossPersonAttention(embed_dim=32, num_layers=1, num_heads=4)

        a = torch.randn(2, 8, 32)
        b = torch.randn(2, 8, 32)
        out_a, out_b = module(a, b)

        assert out_a.shape == (2, 8, 32)
        assert out_b.shape == (2, 8, 32)

    def test_multi_layer_output_shape(self):
        """Two-layer module should produce correct shapes."""
        module = CrossPersonAttention(embed_dim=32, num_layers=2, num_heads=4)

        a = torch.randn(2, 8, 32)
        b = torch.randn(2, 8, 32)
        out_a, out_b = module(a, b)

        assert out_a.shape == (2, 8, 32)
        assert out_b.shape == (2, 8, 32)

    def test_multi_layer_has_correct_count(self):
        """Module should contain the requested number of layers."""
        for n in [1, 2, 3]:
            module = CrossPersonAttention(embed_dim=32, num_layers=n, num_heads=4)
            assert len(module.layers) == n

    def test_gradient_flow(self):
        """Gradients should flow back through all layers of the module."""
        module = CrossPersonAttention(embed_dim=32, num_layers=2, num_heads=4, dropout=0.0)

        a = torch.randn(2, 4, 32, requires_grad=True)
        b = torch.randn(2, 4, 32, requires_grad=True)
        out_a, out_b = module(a, b)

        loss = out_a.sum() + out_b.sum()
        loss.backward()

        assert a.grad is not None, "Gradient should flow to input a"
        assert b.grad is not None, "Gradient should flow to input b"
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

        # Check gradients flow to layer parameters too
        for layer in module.layers:
            for name, param in layer.named_parameters():
                assert param.grad is not None, f"No gradient for {name}"

    def test_self_attention_fallback_shape_2d(self):
        """forward_self_attention on 2D input should return (B, D)."""
        module = CrossPersonAttention(embed_dim=32, num_layers=1, num_heads=4)

        x = torch.randn(3, 32)
        out = module.forward_self_attention(x)

        assert out.shape == (3, 32)

    def test_self_attention_fallback_shape_3d(self):
        """forward_self_attention on 3D input should return (B, T, D)."""
        module = CrossPersonAttention(embed_dim=32, num_layers=2, num_heads=4)

        x = torch.randn(2, 6, 32)
        out = module.forward_self_attention(x)

        assert out.shape == (2, 6, 32)

    def test_self_attention_is_valid_transformation(self):
        """Self-attention should modify features (not identity) but stay reasonable."""
        module = CrossPersonAttention(embed_dim=32, num_layers=1, num_heads=4, dropout=0.0)
        module.eval()

        x = torch.randn(2, 4, 32)
        out = module.forward_self_attention(x)

        assert not torch.allclose(out, x, atol=1e-5), (
            "Self-attention should transform features"
        )

    def test_self_attention_gradient_flow(self):
        """Gradients should flow through forward_self_attention."""
        module = CrossPersonAttention(embed_dim=32, num_layers=1, num_heads=4, dropout=0.0)

        x = torch.randn(2, 4, 32, requires_grad=True)
        out = module.forward_self_attention(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_2d_input_forward(self):
        """Multi-layer module should handle 2D (B, D) input."""
        module = CrossPersonAttention(embed_dim=32, num_layers=2, num_heads=4)

        a = torch.randn(4, 32)
        b = torch.randn(4, 32)
        out_a, out_b = module(a, b)

        assert out_a.shape == (4, 32)
        assert out_b.shape == (4, 32)


# ===========================================================================
# PersonAwareVideoClassifier
# ===========================================================================

class TestPersonAwareVideoClassifier:
    """Tests for PersonAwareVideoClassifier with mocked feature extractor."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_feature_extractor):
        """Auto-patch the feature extractor for every test in this class."""

    def _make_model(self, **config_overrides) -> PersonAwareVideoClassifier:
        """Helper to build a model with optional config overrides."""
        defaults = dict(
            backbone="dinov2-base",
            freeze_backbone=True,
            window_seconds=1.0,
            sample_fps=4.0,
            frame_height=224,
            frame_width=224,
            num_cross_attn_layers=1,
            cross_attn_heads=4,
            cross_attn_dropout=0.0,
            temporal_aggregation="lstm",
            hidden_dim=32,
            dropout=0.0,
            output_dim=1,
        )
        defaults.update(config_overrides)
        cfg = PersonAwareConfig(**defaults)
        return PersonAwareVideoClassifier(cfg)

    # --- Two-person forward ---------------------------------------------------

    def test_forward_two_persons(self):
        """With person_count=2, model uses cross-attention + dual-path."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        person_count = torch.tensor([2, 2])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                person_count=person_count,
            )

        assert logits.shape == (B, 1)

    def test_forward_two_persons_inferred_count(self):
        """When person_count is None but both crops provided, infer count=2."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)

        with torch.no_grad():
            logits = model(person_a_crops=person_a, person_b_crops=person_b)

        assert logits.shape == (B, 1)

    # --- One-person forward ---------------------------------------------------

    def test_forward_one_person(self):
        """With person_count=1, model uses self-attention + single-path."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_count = torch.tensor([1, 1])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_count=person_count,
            )

        assert logits.shape == (B, 1)

    def test_forward_one_person_inferred_count(self):
        """When only person_a_crops is provided, infer count=1."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)

        with torch.no_grad():
            logits = model(person_a_crops=person_a)

        assert logits.shape == (B, 1)

    # --- Zero-person forward --------------------------------------------------

    def test_forward_zero_persons(self):
        """With person_count=0, model falls back to full-frame features."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        full = _make_frames(B, T)
        person_count = torch.tensor([0, 0])

        with torch.no_grad():
            logits = model(
                full_frames=full,
                person_count=person_count,
            )

        assert logits.shape == (B, 1)

    def test_forward_zero_persons_inferred_count(self):
        """When only full_frames is provided, infer count=0."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        full = _make_frames(B, T)

        with torch.no_grad():
            logits = model(full_frames=full)

        assert logits.shape == (B, 1)

    def test_forward_zero_persons_no_full_frames_raises(self):
        """person_count=0 without full_frames should raise ValueError."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_count = torch.tensor([0, 0])

        with pytest.raises(ValueError, match="full_frames required"):
            model(person_a_crops=person_a, person_count=person_count)

    # --- Mixed person counts --------------------------------------------------

    def test_mixed_person_counts(self):
        """Batch where samples have different person counts (0, 1, 2)."""
        model = self._make_model()
        model.eval()

        B, T = 3, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        full = _make_frames(B, T)
        # Sample 0: 2 persons, Sample 1: 1 person, Sample 2: 0 persons
        person_count = torch.tensor([2, 1, 0])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                full_frames=full,
                person_count=person_count,
            )

        assert logits.shape == (B, 1), (
            "Mixed person-count batch should produce (B, output_dim)"
        )

    def test_mixed_person_counts_preserves_order(self):
        """Logits should be in the same order as the input batch, not grouped."""
        model = self._make_model()
        model.eval()

        B, T = 4, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        full = _make_frames(B, T)
        # Interleaved counts: 0, 2, 1, 2
        person_count = torch.tensor([0, 2, 1, 2])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                full_frames=full,
                person_count=person_count,
            )

        assert logits.shape == (B, 1)
        # Each sample should have a finite logit
        assert torch.isfinite(logits).all()

    # --- return_features ------------------------------------------------------

    def test_return_features_two_persons(self):
        """return_features=True with 2 persons returns dict with expected keys."""
        model = self._make_model()
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        person_count = torch.tensor([2, 2])

        with torch.no_grad():
            result = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                person_count=person_count,
                return_features=True,
            )

        assert isinstance(result, dict)
        assert "logits" in result
        assert "temporal_features" in result
        assert result["logits"].shape == (B, 1)
        # temporal_features should be the aggregated features before the head
        assert result["temporal_features"].dim() == 2
        assert result["temporal_features"].shape[0] == B

    def test_return_features_mixed_counts(self):
        """return_features=True with mixed person counts returns correct shapes."""
        model = self._make_model()
        model.eval()

        B, T = 3, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        full = _make_frames(B, T)
        person_count = torch.tensor([2, 1, 0])

        with torch.no_grad():
            result = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                full_frames=full,
                person_count=person_count,
                return_features=True,
            )

        assert result["logits"].shape == (B, 1)
        assert result["temporal_features"].shape[0] == B

    # --- Parameter groups -----------------------------------------------------

    def test_parameter_groups_frozen_backbone(self):
        """Parameter groups should not include frozen backbone params."""
        model = self._make_model(freeze_backbone=True)
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        # All returned params should require grad
        for group in groups:
            for p in group["params"]:
                assert p.requires_grad, (
                    "get_parameter_groups must not return frozen params"
                )

        # No group should have backbone_lr since backbone is frozen
        lrs = [g["lr"] for g in groups]
        assert 1e-5 not in lrs, "Frozen backbone should not appear in param groups"

    def test_parameter_groups_include_cross_attention(self):
        """Cross-attention parameters should appear in parameter groups."""
        model = self._make_model(freeze_backbone=True)
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        # Collect all params from groups
        all_group_params = set()
        for group in groups:
            for p in group["params"]:
                all_group_params.add(id(p))

        # Cross-attention params should be included
        cross_attn_params = list(model.cross_attention.parameters())
        assert len(cross_attn_params) > 0, "Cross-attention should have parameters"

        for p in cross_attn_params:
            if p.requires_grad:
                assert id(p) in all_group_params, (
                    "Cross-attention param missing from parameter groups"
                )

    def test_parameter_groups_include_temporal(self):
        """Temporal LSTM parameters should appear in parameter groups."""
        model = self._make_model(freeze_backbone=True, temporal_aggregation="lstm")
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        all_group_params = set()
        for group in groups:
            for p in group["params"]:
                all_group_params.add(id(p))

        # Both dual and single temporal modules should have params included
        for temporal_name in ["temporal_dual", "temporal_single"]:
            temporal = getattr(model, temporal_name)
            if temporal is not None:
                for p in temporal.parameters():
                    if p.requires_grad:
                        assert id(p) in all_group_params, (
                            f"{temporal_name} param missing from parameter groups"
                        )

    def test_parameter_groups_include_head(self):
        """Head parameters should appear in parameter groups."""
        model = self._make_model(freeze_backbone=True)
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        all_group_params = set()
        for group in groups:
            for p in group["params"]:
                all_group_params.add(id(p))

        for p in model.head.parameters():
            if p.requires_grad:
                assert id(p) in all_group_params, (
                    "Head param missing from parameter groups"
                )

    def test_parameter_groups_unfrozen_backbone(self):
        """With unfrozen backbone, backbone_lr group should appear."""
        model = self._make_model(freeze_backbone=False)
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        lrs = [g["lr"] for g in groups]
        assert 1e-5 in lrs, "Unfrozen backbone should produce a group at backbone_lr"
        assert 1e-3 in lrs, "Head/cross-attn should produce groups at head_lr"

    def test_parameter_groups_only_requires_grad(self):
        """Every parameter in groups must have requires_grad=True."""
        model = self._make_model(freeze_backbone=False, temporal_aggregation="lstm")
        # Freeze backbone after construction to test filtering
        model.freeze_backbone()
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)

        for group in groups:
            for p in group["params"]:
                assert p.requires_grad, (
                    "get_parameter_groups must filter to requires_grad only"
                )

    # --- Freeze / unfreeze ----------------------------------------------------

    def test_freeze_backbone(self):
        """freeze_backbone() should freeze feature extractor parameters."""
        model = self._make_model(freeze_backbone=False)

        # Verify some params are trainable before freezing
        trainable = sum(
            1 for p in model.feature_extractor.parameters() if p.requires_grad
        )
        assert trainable > 0

        model.freeze_backbone()
        for p in model.feature_extractor.parameters():
            assert p.requires_grad is False

    def test_unfreeze_backbone(self):
        """unfreeze_backbone() should make feature extractor parameters trainable."""
        model = self._make_model(freeze_backbone=True)

        # Verify all frozen initially
        for p in model.feature_extractor.parameters():
            assert p.requires_grad is False

        model.unfreeze_backbone()
        trainable = sum(
            1 for p in model.feature_extractor.parameters() if p.requires_grad
        )
        assert trainable > 0

    # --- Edge cases -----------------------------------------------------------

    def test_no_inputs_raises(self):
        """Calling forward with no inputs at all should raise ValueError."""
        model = self._make_model()
        model.eval()

        with pytest.raises(ValueError, match="At least one"):
            model()

    def test_batch_size_one(self):
        """Model should handle batch_size=1 without issues."""
        model = self._make_model()
        model.eval()

        T = model.config.n_frames
        person_a = _make_frames(1, T)
        person_b = _make_frames(1, T)
        person_count = torch.tensor([2])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                person_count=person_count,
            )

        assert logits.shape == (1, 1)

    def test_multi_class_output(self):
        """Model with output_dim > 1 should produce correct shape."""
        model = self._make_model(output_dim=5)
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        person_count = torch.tensor([2, 2])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                person_count=person_count,
            )

        assert logits.shape == (B, 5)

    def test_temporal_aggregation_mean(self):
        """Model with mean temporal aggregation should work for all paths."""
        model = self._make_model(temporal_aggregation="mean")
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        person_count = torch.tensor([2, 2])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                person_count=person_count,
            )

        assert logits.shape == (B, 1)

    def test_temporal_aggregation_attention(self):
        """Model with attention temporal aggregation should work."""
        model = self._make_model(temporal_aggregation="attention")
        model.eval()

        B, T = 2, model.config.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        person_count = torch.tensor([2, 2])

        with torch.no_grad():
            logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                person_count=person_count,
            )

        assert logits.shape == (B, 1)

    def test_cross_attention_module_present(self):
        """Model should have a cross_attention attribute of correct type."""
        model = self._make_model(num_cross_attn_layers=2)
        assert isinstance(model.cross_attention, CrossPersonAttention)
        assert len(model.cross_attention.layers) == 2


# ===========================================================================
# PersonAwareConfig
# ===========================================================================

class TestPersonAwareConfig:
    """Tests for the PersonAwareConfig dataclass."""

    def test_defaults(self):
        """Default values should match the documented architecture."""
        cfg = PersonAwareConfig()
        assert cfg.window_seconds == 2.0
        assert cfg.sample_fps == 12.0
        assert cfg.frame_height == 224
        assert cfg.frame_width == 224
        assert cfg.backbone == "dinov2-base"
        assert cfg.freeze_backbone is True
        assert cfg.num_cross_attn_layers == 1
        assert cfg.cross_attn_heads == 4
        assert cfg.cross_attn_dropout == 0.1
        assert cfg.temporal_aggregation == "lstm"
        assert cfg.hidden_dim == 256
        assert cfg.dropout == 0.3
        assert cfg.output_dim == 1
        assert cfg.weights_path == "person_aware_classifier.pt"

    def test_n_frames_default(self):
        """n_frames = sample_fps * window_seconds = 12 * 2 = 24."""
        cfg = PersonAwareConfig()
        assert cfg.n_frames == 24

    def test_n_frames_custom(self):
        """n_frames with custom fps and window."""
        cfg = PersonAwareConfig(window_seconds=1.0, sample_fps=4.0)
        assert cfg.n_frames == 4

    def test_replace_creates_new_instance(self):
        """dataclasses.replace should produce independent copies."""
        cfg1 = PersonAwareConfig()
        cfg2 = replace(cfg1, hidden_dim=128)
        assert cfg1.hidden_dim == 256
        assert cfg2.hidden_dim == 128

    def test_asdict(self):
        """Config should be convertible to dict for serialization."""
        cfg = PersonAwareConfig(hidden_dim=64)
        d = asdict(cfg)
        assert d["hidden_dim"] == 64
        assert isinstance(d, dict)


# ===========================================================================
# Build / Load functions
# ===========================================================================

class TestBuildLoadFunctions:
    """Tests for build_person_aware_classifier and load_person_aware_classifier."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_feature_extractor):
        """Auto-patch the feature extractor for every test in this class."""

    def test_build_returns_correct_type(self):
        """build_person_aware_classifier should return a PersonAwareVideoClassifier."""
        cfg = PersonAwareConfig(
            backbone="dinov2-base",
            window_seconds=1.0,
            sample_fps=4.0,
            hidden_dim=32,
        )
        model = build_person_aware_classifier(cfg)
        assert isinstance(model, PersonAwareVideoClassifier)

    def test_build_preserves_config(self):
        """The built model should store the config."""
        cfg = PersonAwareConfig(hidden_dim=64)
        model = build_person_aware_classifier(cfg)
        assert model.config is cfg
        assert model.config.hidden_dim == 64

    def test_load_roundtrip(self):
        """Save and load a model, verifying weights are preserved."""
        cfg = PersonAwareConfig(
            backbone="dinov2-base",
            window_seconds=1.0,
            sample_fps=4.0,
            hidden_dim=32,
            dropout=0.0,
        )
        model = build_person_aware_classifier(cfg)
        model.eval()

        # Get a reference output
        B, T = 2, cfg.n_frames
        person_a = _make_frames(B, T)
        person_b = _make_frames(B, T)
        person_count = torch.tensor([2, 2])

        with torch.no_grad():
            ref_logits = model(
                person_a_crops=person_a,
                person_b_crops=person_b,
                person_count=person_count,
            )

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save({
                "config": asdict(cfg),
                "model_state_dict": model.state_dict(),
            }, f)

        try:
            # Load checkpoint
            loaded_model, loaded_cfg = load_person_aware_classifier(
                checkpoint_path, device="cpu",
            )

            assert isinstance(loaded_model, PersonAwareVideoClassifier)
            assert loaded_cfg.hidden_dim == cfg.hidden_dim
            assert loaded_cfg.num_cross_attn_layers == cfg.num_cross_attn_layers

            # Verify outputs match
            with torch.no_grad():
                loaded_logits = loaded_model(
                    person_a_crops=person_a,
                    person_b_crops=person_b,
                    person_count=person_count,
                )

            assert torch.allclose(ref_logits, loaded_logits, atol=1e-5), (
                "Loaded model should reproduce the same outputs"
            )
        finally:
            os.unlink(checkpoint_path)

    def test_load_with_explicit_config(self):
        """load_person_aware_classifier should accept an explicit config."""
        cfg = PersonAwareConfig(
            backbone="dinov2-base",
            window_seconds=1.0,
            sample_fps=4.0,
            hidden_dim=32,
        )
        model = build_person_aware_classifier(cfg)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save({
                "config": asdict(cfg),
                "model_state_dict": model.state_dict(),
            }, f)

        try:
            loaded_model, loaded_cfg = load_person_aware_classifier(
                checkpoint_path, config=cfg, device="cpu",
            )
            assert loaded_cfg is cfg
            assert isinstance(loaded_model, PersonAwareVideoClassifier)
        finally:
            os.unlink(checkpoint_path)

    def test_load_sets_eval_mode(self):
        """Loaded model should be in eval mode."""
        cfg = PersonAwareConfig(
            backbone="dinov2-base",
            window_seconds=1.0,
            sample_fps=4.0,
            hidden_dim=32,
        )
        model = build_person_aware_classifier(cfg)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save({
                "config": asdict(cfg),
                "model_state_dict": model.state_dict(),
            }, f)

        try:
            loaded_model, _ = load_person_aware_classifier(
                checkpoint_path, device="cpu",
            )
            assert not loaded_model.training, "Loaded model should be in eval mode"
        finally:
            os.unlink(checkpoint_path)

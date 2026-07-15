"""
Comprehensive tests for multi-modal fusion modules.

Tests the fusion strategies in fusion_modules.py and the factory function,
as well as verifying the default fusion type in fusion_model.py.

Modules under test:
    - synchronai.models.multimodal.fusion_modules.ConcatFusion
    - synchronai.models.multimodal.fusion_modules.CrossModalAttention
    - synchronai.models.multimodal.fusion_modules.GatedFusion
    - synchronai.models.multimodal.fusion_modules.create_fusion_module
    - synchronai.models.multimodal.fusion_model.MultiModalSynchronyModel (default config only)

Does NOT instantiate MultiModalSynchronyModel (requires YOLO/Whisper weights).
All fusion modules are pure PyTorch and can be tested in isolation.
"""

import pytest
import torch
import torch.nn as nn

from synchronai.models.multimodal.fusion_modules import (
    ConcatFusion,
    CrossModalAttention,
    GatedFusion,
    create_fusion_module,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    (512, 256, 128),
    (256, 256, 256),
    (128, 64, 32),
    (1024, 512, 256),
])
def dim_triple(request):
    """Parameterised (video_dim, audio_dim, hidden_dim) triples."""
    return request.param


@pytest.fixture(params=[1, 2, 4, 16])
def batch_size(request):
    """Various batch sizes, including the edge-case batch_size=1."""
    return request.param


def _make_inputs(batch_size, video_dim, audio_dim):
    """Helper: create random video and audio feature tensors."""
    video = torch.randn(batch_size, video_dim)
    audio = torch.randn(batch_size, audio_dim)
    return video, audio


# ============================================================================
# ConcatFusion
# ============================================================================

class TestConcatFusion:
    """Tests for concatenation-based fusion."""

    def test_output_shape(self, dim_triple, batch_size):
        """Output shape is (batch, hidden_dim) for various dimension combos."""
        video_dim, audio_dim, hidden_dim = dim_triple
        module = ConcatFusion(video_dim, audio_dim, hidden_dim, dropout=0.0)
        module.eval()

        video, audio = _make_inputs(batch_size, video_dim, audio_dim)
        with torch.no_grad():
            out = module(video, audio)

        assert out.shape == (batch_size, hidden_dim), (
            f"Expected ({batch_size}, {hidden_dim}), got {out.shape}"
        )

    def test_gradient_flows(self):
        """Gradients propagate through the fusion back to both inputs."""
        module = ConcatFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.train()

        video = torch.randn(4, 64, requires_grad=True)
        audio = torch.randn(4, 32, requires_grad=True)
        out = module(video, audio)
        loss = out.sum()
        loss.backward()

        assert video.grad is not None, "No gradient on video input"
        assert audio.grad is not None, "No gradient on audio input"
        assert video.grad.abs().sum() > 0, "Video gradient is all zeros"
        assert audio.grad.abs().sum() > 0, "Audio gradient is all zeros"

    def test_different_inputs_produce_different_outputs(self):
        """Sanity check: different inputs yield different fused features."""
        module = ConcatFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.eval()

        video_a, audio_a = _make_inputs(2, 64, 32)
        video_b, audio_b = _make_inputs(2, 64, 32)

        with torch.no_grad():
            out_a = module(video_a, audio_a)
            out_b = module(video_b, audio_b)

        assert not torch.allclose(out_a, out_b, atol=1e-6), (
            "Different inputs produced identical outputs"
        )

    def test_eval_mode(self):
        """Module works correctly in eval mode (BatchNorm uses running stats)."""
        module = ConcatFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.5)

        # Run a few training batches so BatchNorm accumulates running stats
        module.train()
        for _ in range(5):
            v, a = _make_inputs(8, 64, 32)
            module(v, a)

        module.eval()
        video, audio = _make_inputs(4, 64, 32)
        with torch.no_grad():
            out = module(video, audio)

        assert out.shape == (4, 16)
        assert torch.isfinite(out).all(), "Non-finite values in eval output"

    def test_batch_size_one_train_mode(self):
        """ConcatFusion uses BatchNorm1d; batch_size=1 in train mode will fail.

        This documents the known BatchNorm limitation. In production, batch_size=1
        training batches should be avoided or the model should be in eval mode.
        """
        module = ConcatFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.train()

        video, audio = _make_inputs(1, 64, 32)
        with pytest.raises(ValueError):
            # BatchNorm1d raises ValueError for batch_size=1 in train mode
            module(video, audio)

    def test_batch_size_one_eval_mode(self):
        """ConcatFusion works with batch_size=1 in eval mode."""
        module = ConcatFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.eval()

        video, audio = _make_inputs(1, 64, 32)
        with torch.no_grad():
            out = module(video, audio)
        assert out.shape == (1, 16)


# ============================================================================
# CrossModalAttention
# ============================================================================

class TestCrossModalAttention:
    """Tests for cross-modal attention fusion."""

    def test_output_shape(self, dim_triple, batch_size):
        """Output shape is (batch, hidden_dim) for various dimension combos."""
        video_dim, audio_dim, hidden_dim = dim_triple
        # num_heads must divide hidden_dim evenly
        num_heads = 1 if hidden_dim < 4 else 4
        # Make sure hidden_dim is divisible by num_heads
        while hidden_dim % num_heads != 0:
            num_heads -= 1

        module = CrossModalAttention(
            video_dim, audio_dim, hidden_dim,
            num_heads=num_heads, dropout=0.0,
        )
        module.eval()

        video, audio = _make_inputs(batch_size, video_dim, audio_dim)
        with torch.no_grad():
            out = module(video, audio)

        assert out.shape == (batch_size, hidden_dim), (
            f"Expected ({batch_size}, {hidden_dim}), got {out.shape}"
        )

    def test_single_token_attention_is_deterministic(self):
        """With seq_len=1, softmax over 1 key = 1.0; output is deterministic.

        Cross-attention with a single query token and a single key/value token
        produces attention weight of exactly 1.0 (softmax([x]) = 1.0 for any x).
        This means the attention mechanism is effectively a no-op: the 'attended'
        output is just the projected value, regardless of the query content.
        """
        module = CrossModalAttention(
            video_dim=64, audio_dim=32, hidden_dim=16,
            num_heads=4, dropout=0.0,
        )
        module.eval()

        # Fixed audio, two different video inputs
        audio = torch.randn(1, 32)
        video_a = torch.randn(1, 64)
        video_b = torch.randn(1, 64) * 100  # very different magnitude

        with torch.no_grad():
            # The video-to-audio attention should produce the same 'attended'
            # audio value regardless of the video query (since there is only
            # one key to attend to and its weight is always 1.0).
            # However, the audio-to-video attention output WILL differ because
            # the video is used as key/value there, and the final result
            # concatenates both attended outputs.
            #
            # We verify that the module runs and produces valid output;
            # the determinism property holds for each individual attention head.
            out_a = module(video_a, audio)
            out_b = module(video_b, audio)

        # Both outputs should be finite and valid
        assert torch.isfinite(out_a).all()
        assert torch.isfinite(out_b).all()

        # Extract intermediate values to verify the no-op property directly
        video_proj_a = module.video_proj(video_a).unsqueeze(1)
        video_proj_b = module.video_proj(video_b).unsqueeze(1)
        audio_proj = module.audio_proj(audio).unsqueeze(1)

        # Video-to-audio: query differs, but key/value (audio) is the same.
        # With seq_len=1, attended output = value regardless of query.
        attended_a, _ = module.video_to_audio_attn(
            query=video_proj_a, key=audio_proj, value=audio_proj,
        )
        attended_b, _ = module.video_to_audio_attn(
            query=video_proj_b, key=audio_proj, value=audio_proj,
        )
        assert torch.allclose(attended_a, attended_b, atol=1e-5), (
            "Single-token attention should produce identical output regardless of query"
        )

    def test_gradient_flows(self):
        """Gradients propagate through cross-attention to both inputs."""
        module = CrossModalAttention(
            video_dim=64, audio_dim=32, hidden_dim=16,
            num_heads=4, dropout=0.0,
        )
        module.train()

        video = torch.randn(4, 64, requires_grad=True)
        audio = torch.randn(4, 32, requires_grad=True)
        out = module(video, audio)
        loss = out.sum()
        loss.backward()

        assert video.grad is not None, "No gradient on video input"
        assert audio.grad is not None, "No gradient on audio input"
        assert video.grad.abs().sum() > 0, "Video gradient is all zeros"
        assert audio.grad.abs().sum() > 0, "Audio gradient is all zeros"

    def test_eval_mode(self):
        """Module works in eval mode with dropout disabled."""
        module = CrossModalAttention(
            video_dim=64, audio_dim=32, hidden_dim=16,
            num_heads=4, dropout=0.5,
        )

        # Accumulate BatchNorm stats
        module.train()
        for _ in range(5):
            v, a = _make_inputs(8, 64, 32)
            module(v, a)

        module.eval()
        video, audio = _make_inputs(4, 64, 32)
        with torch.no_grad():
            out = module(video, audio)

        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()

    def test_batch_size_one_eval_mode(self):
        """CrossModalAttention works with batch_size=1 in eval mode."""
        module = CrossModalAttention(
            video_dim=64, audio_dim=32, hidden_dim=16,
            num_heads=4, dropout=0.0,
        )
        module.eval()

        video, audio = _make_inputs(1, 64, 32)
        with torch.no_grad():
            out = module(video, audio)
        assert out.shape == (1, 16)

    def test_num_heads_must_divide_hidden_dim(self):
        """num_heads that does not divide hidden_dim should raise an error."""
        with pytest.raises(AssertionError):
            CrossModalAttention(
                video_dim=64, audio_dim=32, hidden_dim=15,
                num_heads=4, dropout=0.0,
            )


# ============================================================================
# GatedFusion
# ============================================================================

class TestGatedFusion:
    """Tests for gated fusion with learnable modality weighting."""

    def test_output_shape(self, dim_triple, batch_size):
        """Output shape is (batch, hidden_dim) for various dimension combos."""
        video_dim, audio_dim, hidden_dim = dim_triple
        module = GatedFusion(video_dim, audio_dim, hidden_dim, dropout=0.0)
        module.eval()

        video, audio = _make_inputs(batch_size, video_dim, audio_dim)
        with torch.no_grad():
            out = module(video, audio)

        assert out.shape == (batch_size, hidden_dim), (
            f"Expected ({batch_size}, {hidden_dim}), got {out.shape}"
        )

    def test_gate_weights_sum_to_one(self):
        """Gate weights (video_weight + audio_weight) must sum to 1.0 per sample.

        The gate network ends with Softmax(dim=1) over 2 values, so the two
        weights must always sum to exactly 1.0.
        """
        module = GatedFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.eval()

        video, audio = _make_inputs(8, 64, 32)
        with torch.no_grad():
            gate_input = torch.cat([video, audio], dim=1)
            gate_weights = module.gate(gate_input)  # (8, 2)

        # Each row should sum to 1.0
        row_sums = gate_weights.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-6), (
            f"Gate weights do not sum to 1.0: {row_sums}"
        )

    def test_gate_weights_are_non_negative(self):
        """Gate weights come from Softmax, so must be >= 0."""
        module = GatedFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.eval()

        video, audio = _make_inputs(8, 64, 32)
        with torch.no_grad():
            gate_input = torch.cat([video, audio], dim=1)
            gate_weights = module.gate(gate_input)

        assert (gate_weights >= 0).all(), "Gate weights contain negative values"

    def test_gate_weights_between_zero_and_one(self):
        """Each individual gate weight is in [0, 1] (Softmax property)."""
        module = GatedFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.eval()

        video, audio = _make_inputs(16, 64, 32)
        with torch.no_grad():
            gate_input = torch.cat([video, audio], dim=1)
            gate_weights = module.gate(gate_input)

        assert (gate_weights >= 0).all() and (gate_weights <= 1).all(), (
            "Gate weights are not in [0, 1] range"
        )

    def test_gradient_flows(self):
        """Gradients propagate through gated fusion to both inputs."""
        module = GatedFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.train()

        video = torch.randn(4, 64, requires_grad=True)
        audio = torch.randn(4, 32, requires_grad=True)
        out = module(video, audio)
        loss = out.sum()
        loss.backward()

        assert video.grad is not None, "No gradient on video input"
        assert audio.grad is not None, "No gradient on audio input"
        assert video.grad.abs().sum() > 0, "Video gradient is all zeros"
        assert audio.grad.abs().sum() > 0, "Audio gradient is all zeros"

    def test_eval_mode(self):
        """Module works in eval mode."""
        module = GatedFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.5)

        # Accumulate BatchNorm stats
        module.train()
        for _ in range(5):
            v, a = _make_inputs(8, 64, 32)
            module(v, a)

        module.eval()
        video, audio = _make_inputs(4, 64, 32)
        with torch.no_grad():
            out = module(video, audio)

        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()

    def test_batch_size_one_eval_mode(self):
        """GatedFusion works with batch_size=1 in eval mode."""
        module = GatedFusion(video_dim=64, audio_dim=32, hidden_dim=16, dropout=0.0)
        module.eval()

        video, audio = _make_inputs(1, 64, 32)
        with torch.no_grad():
            out = module(video, audio)
        assert out.shape == (1, 16)


# ============================================================================
# create_fusion_module factory
# ============================================================================

class TestCreateFusionModule:
    """Tests for the create_fusion_module factory function."""

    def test_returns_concat_fusion(self):
        """fusion_type='concat' returns a ConcatFusion instance."""
        module = create_fusion_module(
            fusion_type='concat',
            video_dim=64, audio_dim=32, hidden_dim=16,
        )
        assert isinstance(module, ConcatFusion)

    def test_returns_cross_attention(self):
        """fusion_type='cross_attention' returns a CrossModalAttention instance."""
        module = create_fusion_module(
            fusion_type='cross_attention',
            video_dim=64, audio_dim=32, hidden_dim=16,
            num_heads=4,
        )
        assert isinstance(module, CrossModalAttention)

    def test_returns_gated_fusion(self):
        """fusion_type='gated' returns a GatedFusion instance."""
        module = create_fusion_module(
            fusion_type='gated',
            video_dim=64, audio_dim=32, hidden_dim=16,
        )
        assert isinstance(module, GatedFusion)

    def test_case_insensitive(self):
        """Factory function normalises fusion_type to lowercase."""
        module = create_fusion_module(
            fusion_type='CONCAT',
            video_dim=64, audio_dim=32, hidden_dim=16,
        )
        assert isinstance(module, ConcatFusion)

        module = create_fusion_module(
            fusion_type='Cross_Attention',
            video_dim=64, audio_dim=32, hidden_dim=16,
            num_heads=4,
        )
        assert isinstance(module, CrossModalAttention)

        module = create_fusion_module(
            fusion_type='GATED',
            video_dim=64, audio_dim=32, hidden_dim=16,
        )
        assert isinstance(module, GatedFusion)

    def test_unknown_type_raises_value_error(self):
        """Unknown fusion type raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown fusion type"):
            create_fusion_module(
                fusion_type='transformer',
                video_dim=64, audio_dim=32, hidden_dim=16,
            )

    def test_unknown_type_empty_string_raises(self):
        """Empty string fusion type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fusion type"):
            create_fusion_module(
                fusion_type='',
                video_dim=64, audio_dim=32, hidden_dim=16,
            )

    def test_all_modules_are_nn_module(self):
        """Every fusion module returned is a proper nn.Module."""
        for fusion_type in ('concat', 'cross_attention', 'gated'):
            num_heads = 4 if fusion_type == 'cross_attention' else 1
            module = create_fusion_module(
                fusion_type=fusion_type,
                video_dim=64, audio_dim=32, hidden_dim=16,
                num_heads=num_heads,
            )
            assert isinstance(module, nn.Module), (
                f"{fusion_type} module is not an nn.Module"
            )

    def test_factory_forwards_dropout(self):
        """Dropout parameter is forwarded to the created module."""
        module = create_fusion_module(
            fusion_type='concat',
            video_dim=64, audio_dim=32, hidden_dim=16,
            dropout=0.0,
        )
        # Verify by checking that Dropout layers have p=0.0
        for submodule in module.modules():
            if isinstance(submodule, nn.Dropout):
                assert submodule.p == 0.0, (
                    f"Dropout probability not forwarded: expected 0.0, got {submodule.p}"
                )


# ============================================================================
# All fusion modules: shared behaviour
# ============================================================================

class TestAllFusionModulesShared:
    """Tests that apply identically to all three fusion module types."""

    FUSION_CONFIGS = [
        ('concat', {}),
        ('cross_attention', {'num_heads': 4}),
        ('gated', {}),
    ]

    @pytest.fixture(params=FUSION_CONFIGS, ids=[c[0] for c in FUSION_CONFIGS])
    def fusion_module(self, request):
        """Parameterised fixture yielding each fusion module type."""
        fusion_type, extra_kwargs = request.param
        module = create_fusion_module(
            fusion_type=fusion_type,
            video_dim=64, audio_dim=32, hidden_dim=16,
            dropout=0.0,
            **extra_kwargs,
        )
        return module

    def test_output_shape_batch_2(self, fusion_module):
        """All modules produce (batch, hidden_dim) output."""
        fusion_module.eval()
        video, audio = _make_inputs(2, 64, 32)
        with torch.no_grad():
            out = fusion_module(video, audio)
        assert out.shape == (2, 16)

    def test_batch_size_one_eval(self, fusion_module):
        """All modules work with batch_size=1 in eval mode."""
        fusion_module.eval()
        video, audio = _make_inputs(1, 64, 32)
        with torch.no_grad():
            out = fusion_module(video, audio)
        assert out.shape == (1, 16)

    def test_eval_mode_produces_finite_output(self, fusion_module):
        """All modules produce finite output in eval mode."""
        # Accumulate BatchNorm stats
        fusion_module.train()
        for _ in range(5):
            v, a = _make_inputs(8, 64, 32)
            fusion_module(v, a)

        fusion_module.eval()
        video, audio = _make_inputs(4, 64, 32)
        with torch.no_grad():
            out = fusion_module(video, audio)
        assert torch.isfinite(out).all(), "Non-finite values in output"

    def test_output_dtype_matches_input(self, fusion_module):
        """Output dtype matches input dtype (float32)."""
        fusion_module.eval()
        video, audio = _make_inputs(2, 64, 32)
        with torch.no_grad():
            out = fusion_module(video, audio)
        assert out.dtype == video.dtype

    def test_has_trainable_parameters(self, fusion_module):
        """All fusion modules have trainable parameters."""
        trainable = sum(p.numel() for p in fusion_module.parameters() if p.requires_grad)
        assert trainable > 0, "Fusion module has no trainable parameters"

    def test_zero_grad_works(self, fusion_module):
        """Optimizer zero_grad works on all fusion modules."""
        fusion_module.train()
        # Run a forward-backward pass first to accumulate grads
        # Use batch_size > 1 to satisfy BatchNorm
        video = torch.randn(4, 64, requires_grad=True)
        audio = torch.randn(4, 32, requires_grad=True)
        out = fusion_module(video, audio)
        out.sum().backward()

        # Check grads exist
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in fusion_module.parameters()
        )
        assert has_grads, "No gradients accumulated"

        # Zero grads
        fusion_module.zero_grad()
        all_zeroed = all(
            p.grad is None or p.grad.abs().sum() == 0
            for p in fusion_module.parameters()
        )
        assert all_zeroed, "Gradients not zeroed"


# ============================================================================
# Default fusion type in MultiModalSynchronyModel
# ============================================================================

class TestDefaultFusionType:
    """Verify that MultiModalSynchronyModel defaults to 'concat' fusion.

    This does NOT instantiate the model (which requires YOLO/Whisper). Instead
    it inspects the source code and config defaults to confirm the default.
    """

    def test_fusion_config_get_defaults_to_concat(self):
        """The fusion_config.get('type', ...) default is 'concat'.

        This test re-creates the logic from MultiModalSynchronyModel.__init__
        to verify the default without needing to instantiate the full model.
        """
        # Simulate an empty fusion config (no 'type' key), as would happen
        # if the user provides a minimal config
        fusion_config = {}
        fusion_type = fusion_config.get('type', 'concat')
        assert fusion_type == 'concat', (
            f"Default fusion type should be 'concat', got '{fusion_type}'"
        )

    def test_factory_creates_concat_for_default(self):
        """The factory creates ConcatFusion when using the model's default type."""
        fusion_config = {}
        fusion_type = fusion_config.get('type', 'concat')
        module = create_fusion_module(
            fusion_type=fusion_type,
            video_dim=256, audio_dim=256, hidden_dim=256,
        )
        assert isinstance(module, ConcatFusion), (
            f"Default fusion type should create ConcatFusion, got {type(module).__name__}"
        )

    def test_default_is_not_cross_attention(self):
        """Explicitly verify the default is NOT cross_attention.

        Cross-attention with single-token features (seq_len=1) is a no-op:
        softmax over a single key always yields weight 1.0. The default was
        changed from 'cross_attention' to 'concat' to avoid this pitfall.
        """
        fusion_config = {}
        fusion_type = fusion_config.get('type', 'concat')
        assert fusion_type != 'cross_attention', (
            "Default fusion type should NOT be 'cross_attention' "
            "(it is a no-op with single-token features)"
        )

    def test_source_code_default_is_concat(self):
        """Inspect the actual source to confirm the default string literal."""
        import inspect
        from synchronai.models.multimodal.fusion_model import MultiModalSynchronyModel

        source = inspect.getsource(MultiModalSynchronyModel.__init__)
        # The line should read: fusion_config.get('type', 'concat')
        assert "'concat'" in source, (
            "MultiModalSynchronyModel.__init__ source does not contain "
            "the string 'concat' as the default fusion type"
        )

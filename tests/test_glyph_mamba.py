"""
Innovation #2: Mamba-Glyph Fusion — Unit Tests
================================================

Validates:
  - GlyphCoordinateEmbedding: shape, fallback, sigmalang integration
  - selective_scan_sequential: output shape, causality, D skip connection
  - GlyphSSMLayer: shape, selectivity dims, forward pass
  - GlyphMambaBlock: residual, shape invariance, token_ids=None fallback
  - GlyphMambaModel: end-to-end forward, weight tying, parameter count,
    glyph_selectivity_ratio
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.models.glyph_mamba import (
    GlyphCoordinateEmbedding,
    GlyphMambaBlock,
    GlyphMambaModel,
    GlyphSSMLayer,
    selective_scan_sequential,
)
from src.recycler.glyph_kv_cache import _SIGMALANG_AVAILABLE

sigmalang_required = pytest.mark.skipif(
    not _SIGMALANG_AVAILABLE, reason="sigmalang not installed"
)

# Fixed random seed for reproducibility
torch.manual_seed(42)


# ===========================================================================
# GlyphCoordinateEmbedding
# ===========================================================================

class TestGlyphCoordinateEmbedding:

    def test_output_shape(self):
        emb = GlyphCoordinateEmbedding(d_glyph=32, use_sigmalang=False)
        token_ids = torch.randint(0, 1000, (2, 16))
        out = emb(token_ids)
        assert out.shape == (2, 16, 32)

    def test_output_dtype_float(self):
        emb = GlyphCoordinateEmbedding(d_glyph=16, use_sigmalang=False)
        token_ids = torch.zeros(1, 4, dtype=torch.long)
        out = emb(token_ids)
        assert out.dtype == torch.float32

    def test_fallback_bucketing(self):
        """Without sigmalang: token_id % 256 bucketing."""
        emb = GlyphCoordinateEmbedding(d_glyph=8, use_sigmalang=False)
        # token 0 and 256 should share primitive 0 → identical embedding
        ids_0   = torch.tensor([[0]],   dtype=torch.long)
        ids_256 = torch.tensor([[256]], dtype=torch.long)
        assert torch.allclose(emb(ids_0), emb(ids_256))

    def test_different_tokens_different_embeddings(self):
        emb = GlyphCoordinateEmbedding(d_glyph=32, use_sigmalang=False)
        ids_a = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        ids_b = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        # Very unlikely to be identical after random init
        assert not torch.allclose(emb(ids_a), emb(ids_b))

    def test_batch_consistency(self):
        """Same token in batch dim 0 and 1 → same embedding."""
        emb = GlyphCoordinateEmbedding(d_glyph=16, use_sigmalang=False)
        ids = torch.tensor([[5, 10], [5, 10]], dtype=torch.long)
        out = emb(ids)
        assert torch.allclose(out[0], out[1])

    @sigmalang_required
    def test_sigmalang_tier0_direct_mapping(self):
        """Tier-0 tokens 0-15 map directly to primitive 0-15."""
        emb = GlyphCoordinateEmbedding(d_glyph=8, use_sigmalang=True)
        if emb._mapper is None:
            pytest.skip("sigmalang mapper not loaded")
        # token 5 → primitive 5 directly
        prim = emb.token_to_primitive(torch.tensor([[5]], dtype=torch.long))
        assert prim.item() == 5

    @sigmalang_required
    def test_sigmalang_semantic_dedup(self):
        """Tokens 256 and 384 → same primitive 0x80 → same embedding."""
        emb = GlyphCoordinateEmbedding(d_glyph=16, use_sigmalang=True)
        if emb._mapper is None:
            pytest.skip("sigmalang mapper not loaded")
        ids_256 = torch.tensor([[256]], dtype=torch.long)
        ids_384 = torch.tensor([[384]], dtype=torch.long)
        assert torch.allclose(emb(ids_256), emb(ids_384))

    def test_d_glyph_configurability(self):
        for d in [8, 16, 32, 64]:
            emb = GlyphCoordinateEmbedding(d_glyph=d, use_sigmalang=False)
            ids = torch.randint(0, 100, (1, 8))
            assert emb(ids).shape == (1, 8, d)


# ===========================================================================
# selective_scan_sequential
# ===========================================================================

class TestSelectiveScanSequential:

    def _make_inputs(self, B=2, L=8, d_model=16, d_state=4):
        x     = torch.randn(B, L, d_model)
        delta = torch.abs(torch.randn(B, L, d_model)) + 0.01
        A     = -torch.abs(torch.randn(d_model, d_state))  # must be negative
        B_mat = torch.randn(B, L, d_state)
        C_mat = torch.randn(B, L, d_state)
        D     = torch.ones(d_model)
        return x, delta, A, B_mat, C_mat, D

    def test_output_shape(self):
        B, L, d_model = 2, 8, 16
        x, dt, A, B_mat, C_mat, D = self._make_inputs(B, L, d_model)
        y = selective_scan_sequential(x, dt, A, B_mat, C_mat, D)
        assert y.shape == (B, L, d_model)

    def test_d_skip_connection(self):
        """With A=0 and B=0: y = D * x (pure skip)."""
        B, L, d_model, d_state = 1, 4, 8, 2
        x = torch.ones(B, L, d_model)
        # If delta * A = 0 effectively and B_mat = 0, h_t ≈ 0
        # Then y = C^T h + D*x = D*x
        delta = torch.ones(B, L, d_model) * 0.001  # tiny delta
        A = -torch.ones(d_model, d_state) * 100     # large negative → exp → 0
        B_mat = torch.zeros(B, L, d_state)
        C_mat = torch.zeros(B, L, d_state)
        D = torch.ones(d_model) * 3.0

        y = selective_scan_sequential(x, delta, A, B_mat, C_mat, D)
        expected = x * D.unsqueeze(0).unsqueeze(0)
        assert torch.allclose(y, expected, atol=1e-4), (
            "With B=C=0, y should equal D*x"
        )

    def test_causality_single_batch(self):
        """Output at position t must not depend on inputs at position t+1."""
        B, L, d_model, d_state = 1, 6, 4, 2
        x, dt, A, B_mat, C_mat, D = self._make_inputs(B, L, d_model, d_state)

        y_orig = selective_scan_sequential(x, dt, A, B_mat, C_mat, D)

        # Perturb only the LAST position's input
        x_perturbed = x.clone()
        x_perturbed[0, -1, :] += 10.0

        y_pert = selective_scan_sequential(x_perturbed, dt, A, B_mat, C_mat, D)

        # Outputs at t < L-1 must be unaffected
        assert torch.allclose(y_orig[0, :-1, :], y_pert[0, :-1, :]), (
            "SSM output at t is affected by future input — causality violated"
        )

    def test_positive_A_diverges_warns(self):
        """Positive A causes exponential explosion — not a hard error but values grow."""
        B, L, d_model, d_state = 1, 10, 4, 2
        x = torch.ones(B, L, d_model)
        delta = torch.ones(B, L, d_model)
        A = torch.ones(d_model, d_state) * 2.0  # positive → growing
        B_mat = torch.ones(B, L, d_state) * 0.01
        C_mat = torch.ones(B, L, d_state) * 0.01
        D = torch.zeros(d_model)
        y = selective_scan_sequential(x, delta, A, B_mat, C_mat, D)
        # Values should grow along L (causal explosion)
        norms = [y[0, t].norm().item() for t in range(L)]
        assert norms[-1] > norms[0], "Positive A should cause output to grow"

    def test_batch_independence(self):
        """Changing batch 0's input must not affect batch 1's output."""
        B, L, d_model, d_state = 3, 6, 8, 4
        x, dt, A, B_mat, C_mat, D = self._make_inputs(B, L, d_model, d_state)
        y_orig = selective_scan_sequential(x, dt, A, B_mat, C_mat, D)

        x_mod = x.clone()
        x_mod[0, :, :] = 99.0
        y_mod = selective_scan_sequential(x_mod, dt, A, B_mat, C_mat, D)

        assert torch.allclose(y_orig[1], y_mod[1])
        assert torch.allclose(y_orig[2], y_mod[2])


# ===========================================================================
# GlyphSSMLayer
# ===========================================================================

class TestGlyphSSMLayer:

    def test_output_shape(self):
        layer = GlyphSSMLayer(d_model=64, d_state=8, d_glyph=16)
        B, L = 2, 12
        x = torch.randn(B, L, 64)
        g = torch.randn(B, L, 16)
        y = layer(x, g)
        assert y.shape == (B, L, 64)

    def test_glyph_proj_not_d_model(self):
        """B_proj, C_proj, dt_proj take d_glyph inputs (not d_model)."""
        d_model, d_glyph = 128, 16
        layer = GlyphSSMLayer(d_model=d_model, d_state=8, d_glyph=d_glyph)
        # B_proj: d_glyph → d_state
        assert layer.B_proj.in_features  == d_glyph
        assert layer.C_proj.in_features  == d_glyph
        assert layer.dt_proj_down.in_features == d_glyph
        assert layer.dt_proj_up.out_features  == d_model

    def test_A_always_negative(self):
        """A_log parameterisation must produce strictly negative A."""
        layer = GlyphSSMLayer(d_model=32, d_state=4, d_glyph=8)
        A = -torch.exp(layer.A_log)
        assert (A < 0).all()

    def test_different_glyph_coords_different_output(self):
        """Selectivity: same content x but different glyph coords → different output."""
        layer = GlyphSSMLayer(d_model=32, d_state=4, d_glyph=8)
        B, L = 1, 4
        x = torch.randn(B, L, 32)
        g1 = torch.randn(B, L, 8)
        g2 = torch.randn(B, L, 8)
        y1 = layer(x, g1)
        y2 = layer(x, g2)
        assert not torch.allclose(y1, y2), (
            "Different glyph coords must produce different SSM outputs"
        )

    def test_same_glyph_coords_same_output(self):
        """Determinism: same inputs → same output."""
        layer = GlyphSSMLayer(d_model=32, d_state=4, d_glyph=8)
        layer.eval()
        B, L = 1, 6
        x = torch.randn(B, L, 32)
        g = torch.randn(B, L, 8)
        with torch.no_grad():
            y1 = layer(x, g)
            y2 = layer(x, g)
        assert torch.allclose(y1, y2)

    def test_gradient_flows_through_glyph_proj(self):
        """Backprop should flow through the glyph selectivity projections."""
        layer = GlyphSSMLayer(d_model=16, d_state=4, d_glyph=8)
        B, L = 1, 4
        x = torch.randn(B, L, 16)
        g = torch.randn(B, L, 8, requires_grad=True)
        y = layer(x, g)
        loss = y.sum()
        loss.backward()
        assert g.grad is not None
        assert g.grad.norm() > 0


# ===========================================================================
# GlyphMambaBlock
# ===========================================================================

class TestGlyphMambaBlock:

    def test_output_shape_matches_input(self):
        block = GlyphMambaBlock(d_model=64, d_glyph=16, use_sigmalang=False)
        B, L = 2, 10
        h = torch.randn(B, L, 64)
        token_ids = torch.randint(0, 1000, (B, L))
        out = block(h, token_ids=token_ids)
        assert out.shape == (B, L, 64)

    def test_residual_connection(self):
        """If SSM output is zero, block should output ~ residual."""
        block = GlyphMambaBlock(d_model=32, d_glyph=8, use_sigmalang=False)
        # Zero out the out_proj weights → SSM contribution ≈ 0
        with torch.no_grad():
            block.out_proj.weight.zero_()
        B, L = 1, 4
        h = torch.randn(B, L, 32)
        token_ids = torch.zeros(B, L, dtype=torch.long)
        out = block(h, token_ids=token_ids)
        # Out should be approximately h (residual) since SSM path is zeroed
        assert torch.allclose(out, h, atol=1e-5), (
            "With zeroed out_proj, block output should equal residual"
        )

    def test_no_token_ids_fallback(self):
        """token_ids=None should not crash, uses zero-primitive fallback."""
        block = GlyphMambaBlock(d_model=32, d_glyph=8, use_sigmalang=False)
        B, L = 2, 6
        h = torch.randn(B, L, 32)
        out = block(h, token_ids=None)
        assert out.shape == (B, L, 32)

    def test_conv_strips_extra_padding(self):
        """Output length must equal input length for any L >= 1."""
        block = GlyphMambaBlock(d_model=16, d_glyph=8, use_sigmalang=False)
        for L in [1, 3, 7, 16, 33]:
            h = torch.randn(1, L, 16)
            ids = torch.zeros(1, L, dtype=torch.long)
            out = block(h, token_ids=ids)
            assert out.shape == (1, L, 16), f"Length mismatch at L={L}"

    def test_trainable_parameters(self):
        block = GlyphMambaBlock(d_model=64, d_glyph=16, use_sigmalang=False)
        n_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        assert n_params > 0

    def test_gradient_flows_end_to_end(self):
        block = GlyphMambaBlock(d_model=32, d_glyph=8, use_sigmalang=False)
        B, L = 1, 4
        h = torch.randn(B, L, 32, requires_grad=True)
        token_ids = torch.randint(0, 100, (B, L))
        out = block(h, token_ids=token_ids)
        out.sum().backward()
        assert h.grad is not None

    def test_expand_controls_inner_dim(self):
        """expand=1 vs expand=2 — both should produce correct output shape."""
        for expand in [1, 2, 4]:
            block = GlyphMambaBlock(d_model=32, d_glyph=8, expand=expand, use_sigmalang=False)
            h = torch.randn(1, 4, 32)
            ids = torch.zeros(1, 4, dtype=torch.long)
            assert block(h, token_ids=ids).shape == (1, 4, 32)


# ===========================================================================
# GlyphMambaModel
# ===========================================================================

class TestGlyphMambaModel:

    def test_logits_shape(self):
        model = GlyphMambaModel(vocab_size=256, d_model=32, n_layers=2, use_sigmalang=False)
        token_ids = torch.randint(0, 256, (2, 8))
        logits = model(token_ids)
        assert logits.shape == (2, 8, 256)

    def test_return_hidden(self):
        model = GlyphMambaModel(vocab_size=256, d_model=32, n_layers=2, use_sigmalang=False)
        token_ids = torch.randint(0, 256, (1, 6))
        hidden = model(token_ids, return_hidden=True)
        assert hidden.shape == (1, 6, 32)

    def test_weight_tying(self):
        """lm_head.weight is the same tensor as token_embed.weight."""
        model = GlyphMambaModel(vocab_size=128, d_model=16, n_layers=1, use_sigmalang=False)
        assert model.lm_head.weight is model.token_embed.weight

    def test_count_parameters_positive(self):
        model = GlyphMambaModel(vocab_size=64, d_model=16, n_layers=2, use_sigmalang=False)
        assert model.count_parameters() > 0

    def test_glyph_selectivity_ratio(self):
        """Ratio must be > 0 (glyph params exist) and < 1 (not all params are glyph)."""
        model = GlyphMambaModel(vocab_size=64, d_model=32, n_layers=2, use_sigmalang=False)
        ratio = model.glyph_selectivity_ratio()
        assert 0.0 < ratio < 1.0, f"Unexpected selectivity ratio: {ratio}"

    def test_deeper_model(self):
        """n_layers scaling — 8 layers should still run without error."""
        model = GlyphMambaModel(vocab_size=64, d_model=16, n_layers=8, d_state=4, d_glyph=8, use_sigmalang=False)
        token_ids = torch.randint(0, 64, (1, 12))
        logits = model(token_ids)
        assert logits.shape == (1, 12, 64)

    def test_deterministic_eval(self):
        model = GlyphMambaModel(vocab_size=64, d_model=16, n_layers=2, use_sigmalang=False)
        model.eval()
        token_ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            y1 = model(token_ids)
            y2 = model(token_ids)
        assert torch.allclose(y1, y2)

    def test_gradient_flows_through_all_layers(self):
        model = GlyphMambaModel(vocab_size=64, d_model=16, n_layers=2, use_sigmalang=False)
        token_ids = torch.randint(0, 64, (1, 4))
        logits = model(token_ids)
        loss = logits.sum()
        loss.backward()
        # Check A_log in each block received gradients
        for i, block in enumerate(model.blocks):
            grad = block.ssm.A_log.grad
            assert grad is not None, f"No gradient on A_log in block {i}"

    def test_d_glyph_smaller_than_d_model(self):
        """d_glyph << d_model is the key invariant of the architecture."""
        d_model, d_glyph = 128, 16
        assert d_glyph < d_model
        model = GlyphMambaModel(
            vocab_size=256, d_model=d_model, n_layers=2, d_glyph=d_glyph, use_sigmalang=False
        )
        # Selectivity projections operate in d_glyph space
        for block in model.blocks:
            assert block.ssm.B_proj.in_features  == d_glyph
            assert block.ssm.C_proj.in_features  == d_glyph
            assert block.ssm.B_proj.in_features  <  d_model

    def test_seq_len_one(self):
        """Edge case: single-token sequence should not crash."""
        model = GlyphMambaModel(vocab_size=32, d_model=16, n_layers=1, use_sigmalang=False)
        token_ids = torch.zeros(1, 1, dtype=torch.long)
        logits = model(token_ids)
        assert logits.shape == (1, 1, 32)


# ===========================================================================
# Integration with serving engine
# ===========================================================================

class TestGlyphMambaServingIntegration:

    def test_model_forward_matches_serving_stub_interface(self):
        """
        GlyphMambaModel.forward with return_hidden=True mimics the
        last_hidden_state() interface used by the API server stub.
        """
        model = GlyphMambaModel(vocab_size=256, d_model=32, n_layers=1, use_sigmalang=False)
        model.eval()
        token_ids = torch.randint(0, 256, (1, 10))
        with torch.no_grad():
            hidden = model(token_ids, return_hidden=True)  # [1, 10, 32]
        # Simulate mean-pool as done by the API server
        pooled = hidden.mean(dim=1)  # [1, 32]
        assert pooled.shape == (1, 32)

    def test_glyph_mamba_as_drop_in_backbone(self):
        """
        Replace the stub model in DistributedServingEngine with GlyphMambaModel.
        Engine should initialise without error.
        """
        try:
            from src.serving.distributed_serving import DistributedServingEngine
        except ImportError:
            pytest.skip("distributed_serving not importable")

        model = GlyphMambaModel(vocab_size=256, d_model=32, n_layers=1, use_sigmalang=False)
        # DistributedServingEngine wraps any nn.Module
        with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
            engine = DistributedServingEngine(model, num_gpus=1)
        assert engine is not None

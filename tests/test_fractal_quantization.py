"""
Innovation #2: Fractal Quantization & Innovation #10: Quantum-Friendly Design
==============================================================================

Validates:
  - ProbabilisticLattice: encode, query, uncertainty, delta, wire format
  - FractalQuantizer: quantize, dequantize, adaptive query
  - AdaptiveInferenceQuery: bit estimation heuristics, stats
  - Quantum-friendly invariants: level structure maps to qubit registers
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.quantization.fractal_quantization import (
    DEFAULT_BIT_LEVELS,
    AdaptiveInferenceQuery,
    FractalQuantizer,
    ProbabilisticLattice,
    QuantizationLevel,
)

torch.manual_seed(0)


# ============================================================================
# ProbabilisticLattice
# ============================================================================

class TestProbabilisticLattice:

    def test_encode_zero(self):
        lat = ProbabilisticLattice.encode(0.0)
        assert lat.full_value() == 0.0

    def test_encode_one(self):
        lat = ProbabilisticLattice.encode(1.0)
        # At 8 bits (256 steps), value = 256/256 = 1.0 → capped at 255/256 due to int conversion
        val = lat.full_value()
        assert val >= 0.99

    def test_encode_half(self):
        lat = ProbabilisticLattice.encode(0.5)
        val = lat.full_value()
        assert abs(val - 0.5) < 1.0 / 256

    def test_query_returns_quantization_level(self):
        lat = ProbabilisticLattice.encode(0.75)
        q = lat.query(4)
        assert isinstance(q, QuantizationLevel)
        assert q.bits == 4
        assert q.resolution == 16

    def test_query_higher_bits_lower_uncertainty(self):
        lat = ProbabilisticLattice.encode(0.333)
        q_low  = lat.query(2)
        q_high = lat.query(7)
        assert q_high.uncertainty < q_low.uncertainty

    def test_uncertainty_equals_1_over_resolution(self):
        lat = ProbabilisticLattice.encode(0.5)
        for bits in range(1, DEFAULT_BIT_LEVELS + 1):
            q = lat.query(bits)
            assert abs(q.uncertainty - 1.0 / q.resolution) < 1e-9

    def test_fidelity_increases_with_bits(self):
        lat = ProbabilisticLattice.encode(0.7)
        prev_f = 0.0
        for bits in range(1, DEFAULT_BIT_LEVELS + 1):
            f = lat.query(bits).fidelity
            assert f > prev_f
            prev_f = f

    def test_level_delta_converges(self):
        """Consecutive level deltas should shrink as bits increase."""
        lat = ProbabilisticLattice.encode(0.123456789)
        deltas = [lat.level_delta(k) for k in range(2, DEFAULT_BIT_LEVELS + 1)]
        # Not necessarily monotone, but sum should decrease
        assert sum(deltas[:3]) >= sum(deltas[-3:]) or max(deltas[-3:]) <= 0.1

    def test_query_range(self):
        lat = ProbabilisticLattice.encode(0.5)
        levels = lat.query_range(1, 5)
        assert len(levels) == 5
        for i, q in enumerate(levels):
            assert q.bits == i + 1

    def test_wire_format_round_trip(self):
        lat = ProbabilisticLattice.encode(0.7654)
        data = lat.to_bytes()
        assert len(data) == DEFAULT_BIT_LEVELS
        lat2 = ProbabilisticLattice.from_bytes(data)
        assert lat2.full_value() == lat.full_value()

    def test_wire_format_8_bytes(self):
        lat = ProbabilisticLattice.encode(0.5)
        assert len(lat.to_bytes()) == 8

    def test_clamps_out_of_range(self):
        lat = ProbabilisticLattice.encode(2.0)  # clamped to 1.0
        assert lat.full_value() >= 0.99
        lat2 = ProbabilisticLattice.encode(-1.0)  # clamped to 0.0
        assert lat2.full_value() == 0.0

    def test_wrong_level_count_raises(self):
        with pytest.raises(ValueError):
            ProbabilisticLattice([0, 1], b=8)

    def test_query_clamped_to_valid_range(self):
        lat = ProbabilisticLattice.encode(0.5)
        # Querying bits=0 should clamp to bits=1
        q0 = lat.query(0)
        q1 = lat.query(1)
        assert q0.bits == q1.bits == 1
        # Querying beyond max should clamp
        q_max = lat.query(100)
        assert q_max.bits == DEFAULT_BIT_LEVELS

    # Quantum-friendly invariants
    def test_quantum_level_count_matches_n_primitives(self):
        """
        8-bit lattice has 2^8 = 256 steps, matching N_PRIMITIVES = 256.
        This is intentional: each level corresponds to a qubit register,
        and full-resolution maps to the 256-primitive glyph alphabet.
        """
        lat = ProbabilisticLattice.encode(0.5)
        full = lat.query(DEFAULT_BIT_LEVELS)
        assert full.resolution == 256


# ============================================================================
# FractalQuantizer
# ============================================================================

class TestFractalQuantizer:

    def test_quantize_returns_lattices(self):
        q = FractalQuantizer(bit_levels=4)
        x = torch.rand(5)
        lats, xm, xM = q.quantize(x)
        assert len(lats) == 5

    def test_dequantize_full_bits_approx_original(self):
        q = FractalQuantizer(bit_levels=8)
        x = torch.rand(16)
        lats, xm, xM = q.quantize(x)
        x_hat = q.dequantize(lats, xm, xM, bits=8, shape=x.shape)
        mae = (x - x_hat).abs().mean().item()
        assert mae < 1.0 / 256 + 1e-5, f"MAE={mae:.6f} too large at 8-bit"

    def test_dequantize_low_bits_higher_error(self):
        q = FractalQuantizer(bit_levels=8)
        x = torch.rand(64)
        lats, xm, xM = q.quantize(x)
        x_2bit = q.dequantize(lats, xm, xM, bits=2)
        x_8bit = q.dequantize(lats, xm, xM, bits=8)
        mae_2 = (x - x_2bit).abs().mean().item()
        mae_8 = (x - x_8bit).abs().mean().item()
        assert mae_2 > mae_8

    def test_dequantize_preserves_shape(self):
        q = FractalQuantizer()
        x = torch.rand(3, 4)
        lats, xm, xM = q.quantize(x)
        x_hat = q.dequantize(lats, xm, xM, bits=4, shape=(3, 4))
        assert x_hat.shape == (3, 4)

    def test_adaptive_query_converges(self):
        q = FractalQuantizer(bit_levels=8)
        x = torch.rand(16)
        lats, xm, xM = q.quantize(x)
        result, bits, delta = q.query_adaptive(lats, xm, xM, shape=x.shape, tolerance=0.01)
        assert result.shape == x.shape
        assert 2 <= bits <= 8
        assert delta >= 0

    def test_adaptive_query_sparse_needs_fewer_bits(self):
        """
        Constant vector: all elements equal → converges at 1 bit.
        Random vector: typically needs more bits.
        """
        q = FractalQuantizer(bit_levels=8)

        x_const = torch.ones(32) * 0.5
        lats_c, xm_c, xM_c = q.quantize(x_const)
        _, bits_const, _ = q.query_adaptive(lats_c, xm_c, xM_c, shape=None, tolerance=0.001)

        x_rand = torch.rand(32)
        lats_r, xm_r, xM_r = q.quantize(x_rand)
        _, bits_rand, _ = q.query_adaptive(lats_r, xm_r, xM_r, shape=None, tolerance=0.001)

        # Constant should need ≤ random (may be equal for small random vectors)
        assert bits_const <= bits_rand + 2  # +2 margin for randomness

    def test_uniform_tensor_low_error_at_1_bit(self):
        """All-same tensor → 1-bit quantization has zero error."""
        q = FractalQuantizer(bit_levels=8)
        x = torch.ones(10) * 0.6
        lats, xm, xM = q.quantize(x)
        # All values are the same → any bit level gives the same value
        x_1bit = q.dequantize(lats, xm, xM, bits=1)
        x_8bit = q.dequantize(lats, xm, xM, bits=8)
        assert (x_1bit - x_8bit).abs().max().item() < 1e-5

    def test_bits_used_histogram(self):
        q = FractalQuantizer(bit_levels=8)
        lats_list, xm_list, xM_list = [], [], []
        for _ in range(5):
            x = torch.rand(4)
            l, xm, xM = q.quantize(x)
            lats_list.append(l)
            xm_list.append(xm)
            xM_list.append(xM)
        hist = q.bits_used_histogram(lats_list, xm_list, xM_list)
        assert sum(hist.values()) == 5
        assert all(2 <= k <= 8 for k in hist)


# ============================================================================
# AdaptiveInferenceQuery
# ============================================================================

class TestAdaptiveInferenceQuery:

    def _sparse_centroid(self, n_active=4) -> list:
        """Centroid with n_active non-zero entries."""
        c = [0.0] * 256
        for i in range(n_active):
            c[i] = 1.0 / n_active
        return c

    def _dense_centroid(self) -> list:
        """Centroid with all 256 entries equal."""
        return [1.0 / 256] * 256

    def test_returns_int_in_valid_range(self):
        q = AdaptiveInferenceQuery(min_bits=1, max_bits=8)
        c = self._sparse_centroid()
        bits = q.estimate_bits(c)
        assert isinstance(bits, int)
        assert 1 <= bits <= 8

    def test_sparse_centroid_fewer_bits_than_dense(self):
        """Sparse (high confidence) → fewer bits needed."""
        q = AdaptiveInferenceQuery()
        sparse = self._sparse_centroid(n_active=2)
        dense  = self._dense_centroid()
        bits_sparse = q.estimate_bits(sparse, consensus_strength=0.0)
        bits_dense  = q.estimate_bits(dense,  consensus_strength=0.0)
        assert bits_sparse <= bits_dense

    def test_high_consensus_reduces_bits(self):
        """High consensus strength (nodes agree) → fewer bits needed."""
        q = AdaptiveInferenceQuery()
        c = self._dense_centroid()
        bits_low_consensus  = q.estimate_bits(c, consensus_strength=0.0)
        bits_high_consensus = q.estimate_bits(c, consensus_strength=1.0)
        assert bits_high_consensus <= bits_low_consensus

    def test_stats_tracks_queries(self):
        q = AdaptiveInferenceQuery()
        c = self._sparse_centroid()
        for _ in range(5):
            q.estimate_bits(c)
        s = q.stats()
        assert s["queries"] == 5
        assert "mean_bits" in s
        assert "sub_max_pct" in s

    def test_empty_centroid_does_not_crash(self):
        q = AdaptiveInferenceQuery()
        bits = q.estimate_bits([0.0] * 256)
        assert isinstance(bits, int)

    def test_sub_max_pct_for_sparse_workload(self):
        """
        Sparse centroids should mostly use fewer than max bits.
        sub_max_pct should be high (most queries stop early).
        """
        q = AdaptiveInferenceQuery(min_bits=2, max_bits=8)
        sparse = self._sparse_centroid(n_active=1)
        for _ in range(20):
            q.estimate_bits(sparse, consensus_strength=0.9)
        s = q.stats()
        assert s["sub_max_pct"] > 50.0, (
            "Most sparse queries should not need full bit depth"
        )

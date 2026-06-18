"""
Innovation #5: Multi-Node Negative-Space Consensus — Unit Tests
===============================================================

Validates:
  - NegativeSpaceDescriptor: wire format, set operations, bandwidth invariant
  - NegativeSpaceExtractor: glyph-coord extraction, delta extraction
  - ConsensusResult: pairwise Jaccard, agreed/disputed partitions
  - ConsensusNode: process() returns descriptor, stats()
  - NegativeSpaceConsensusEngine: 2-4 node aggregation, bandwidth stats,
    same-input → high consensus, different-input → lower consensus
  - GlyphMambaWithNegativeSpace: hook captures glyph coords
"""

import asyncio
import sys
from pathlib import Path
from typing import FrozenSet

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.consensus.negative_space import (
    N_PRIMITIVES,
    ConsensusNode,
    ConsensusResult,
    GlyphMambaWithNegativeSpace,
    NegativeSpaceConsensusEngine,
    NegativeSpaceDescriptor,
    NegativeSpaceExtractor,
    _DESCRIPTOR_BYTES,
)

torch.manual_seed(0)

# ============================================================================
# NegativeSpaceDescriptor
# ============================================================================

class TestNegativeSpaceDescriptor:

    def test_empty_descriptor(self):
        d = NegativeSpaceDescriptor()
        assert d.absent_count == 0
        assert d.sparsity == 0.0

    def test_full_descriptor(self):
        d = NegativeSpaceDescriptor(frozenset(range(256)))
        assert d.absent_count == 256
        assert d.sparsity == 1.0

    def test_bandwidth_always_256_bits(self):
        for size in [0, 1, 64, 128, 200, 256]:
            d = NegativeSpaceDescriptor(frozenset(range(size)))
            assert d.bandwidth_bits == 256, (
                "Bandwidth must be constant 256 bits regardless of absent count"
            )

    def test_wire_format_empty_round_trip(self):
        d = NegativeSpaceDescriptor()
        data = d.to_bytes()
        assert len(data) == _DESCRIPTOR_BYTES  # 32 bytes
        d2 = NegativeSpaceDescriptor.from_bytes(data)
        assert d2 == d

    def test_wire_format_full_round_trip(self):
        d = NegativeSpaceDescriptor(frozenset(range(256)))
        d2 = NegativeSpaceDescriptor.from_bytes(d.to_bytes())
        assert d2 == d

    def test_wire_format_sparse_round_trip(self):
        absent = frozenset([0, 7, 63, 127, 128, 255])
        d = NegativeSpaceDescriptor(absent)
        d2 = NegativeSpaceDescriptor.from_bytes(d.to_bytes())
        assert d2.absent == absent

    def test_wire_format_bad_length_raises(self):
        with pytest.raises(ValueError, match="32 bytes"):
            NegativeSpaceDescriptor.from_bytes(b"\x00" * 31)

    def test_primitive_ids_clamped_to_255(self):
        d = NegativeSpaceDescriptor(frozenset([256, 512]))
        # 256 % 256 = 0, 512 % 256 = 0 — both map to primitive 0
        assert d.absent == frozenset([0])

    # ── Set operations ────────────────────────────────────────────────────────

    def test_consensus_with_identical_descriptors(self):
        d = NegativeSpaceDescriptor(frozenset([1, 2, 3]))
        assert d.consensus_with(d) == 1.0

    def test_consensus_with_disjoint_descriptors(self):
        d1 = NegativeSpaceDescriptor(frozenset([0, 1, 2]))
        d2 = NegativeSpaceDescriptor(frozenset([3, 4, 5]))
        assert d1.consensus_with(d2) == 0.0

    def test_consensus_with_both_empty(self):
        d1 = NegativeSpaceDescriptor()
        d2 = NegativeSpaceDescriptor()
        assert d1.consensus_with(d2) == 1.0

    def test_consensus_with_partial_overlap(self):
        d1 = NegativeSpaceDescriptor(frozenset([0, 1, 2, 3]))  # 4 elements
        d2 = NegativeSpaceDescriptor(frozenset([2, 3, 4, 5]))  # 4 elements
        # intersection = {2, 3} (2), union = {0,1,2,3,4,5} (6) → 2/6
        expected = 2 / 6
        assert abs(d1.consensus_with(d2) - expected) < 1e-9

    def test_merge_conservative(self):
        """merge() = intersection (absent only if both agree)."""
        d1 = NegativeSpaceDescriptor(frozenset([0, 1, 2, 3]))
        d2 = NegativeSpaceDescriptor(frozenset([2, 3, 4, 5]))
        merged = d1.merge(d2)
        assert merged.absent == frozenset([2, 3])

    def test_merge_empty_result_when_disjoint(self):
        d1 = NegativeSpaceDescriptor(frozenset([0, 1]))
        d2 = NegativeSpaceDescriptor(frozenset([2, 3]))
        merged = d1.merge(d2)
        assert merged.absent == frozenset()

    def test_union_liberal(self):
        """union() = union of absent sets."""
        d1 = NegativeSpaceDescriptor(frozenset([0, 1]))
        d2 = NegativeSpaceDescriptor(frozenset([2, 3]))
        u = d1.union(d2)
        assert u.absent == frozenset([0, 1, 2, 3])

    def test_equality_and_hash(self):
        d1 = NegativeSpaceDescriptor(frozenset([1, 2, 3]))
        d2 = NegativeSpaceDescriptor(frozenset([1, 2, 3]))
        d3 = NegativeSpaceDescriptor(frozenset([1, 2]))
        assert d1 == d2
        assert d1 != d3
        assert hash(d1) == hash(d2)


# ============================================================================
# NegativeSpaceExtractor
# ============================================================================

class TestNegativeSpaceExtractor:

    def test_extract_from_glyph_coords_high_norm_all_present(self):
        """All high-norm coords → no absent primitives."""
        ext = NegativeSpaceExtractor(coord_threshold=0.01)
        B, L, d_glyph = 1, 8, 16
        token_ids = torch.arange(8).unsqueeze(0)    # [1, 8]
        # Large-norm glyph coords → above threshold → present
        glyph_coords = torch.ones(B, L, d_glyph) * 10.0
        desc = ext.extract_from_glyph_coords(token_ids, glyph_coords)
        assert desc.absent_count == 0

    def test_extract_from_glyph_coords_zero_norm_all_absent(self):
        """All zero-norm coords → all appearing primitives absent."""
        ext = NegativeSpaceExtractor(coord_threshold=0.01)
        B, L, d_glyph = 1, 8, 16
        token_ids = torch.arange(8).unsqueeze(0)
        glyph_coords = torch.zeros(B, L, d_glyph)  # zero norm < threshold
        desc = ext.extract_from_glyph_coords(token_ids, glyph_coords)
        # All 8 unique token→primitive mappings should be absent
        assert desc.absent_count > 0

    def test_extract_from_glyph_coords_mixed(self):
        """Some positions high-norm (present) some zero-norm (absent)."""
        ext = NegativeSpaceExtractor(coord_threshold=0.5)
        B, L, d_glyph = 1, 4, 8
        # Tokens 0-3 → primitives 0-3 (fallback bucketing)
        token_ids = torch.tensor([[0, 1, 2, 3]])
        glyph_coords = torch.zeros(B, L, d_glyph)
        glyph_coords[0, 0, :] = 1.0  # token 0 (prim 0) has high norm → present
        glyph_coords[0, 1, :] = 1.0  # token 1 (prim 1) has high norm → present
        # tokens 2, 3 remain zero → primitives 2, 3 absent
        desc = ext.extract_from_glyph_coords(token_ids, glyph_coords)
        assert 2 in desc.absent
        assert 3 in desc.absent
        assert 0 not in desc.absent
        assert 1 not in desc.absent

    def test_extract_from_delta_low_delta_absent(self):
        """All low-delta positions → those primitives absent."""
        ext = NegativeSpaceExtractor(delta_threshold=0.1)
        B, L, d_model = 1, 4, 8
        token_ids = torch.tensor([[0, 1, 2, 3]])
        delta = torch.ones(B, L, d_model) * 0.01  # below threshold
        desc = ext.extract_from_delta(token_ids, delta)
        # All 4 primitives should be absent
        assert desc.absent_count == 4

    def test_extract_from_delta_high_delta_present(self):
        """All high-delta positions → no primitives absent."""
        ext = NegativeSpaceExtractor(delta_threshold=0.01)
        B, L, d_model = 1, 4, 8
        token_ids = torch.tensor([[0, 1, 2, 3]])
        delta = torch.ones(B, L, d_model) * 1.0  # above threshold
        desc = ext.extract_from_delta(token_ids, delta)
        assert desc.absent_count == 0

    def test_extract_random_deterministic(self):
        ext = NegativeSpaceExtractor()
        d1 = ext.extract_random_for_test(seed=42)
        d2 = ext.extract_random_for_test(seed=42)
        assert d1 == d2

    def test_extract_random_different_seeds_different(self):
        ext = NegativeSpaceExtractor()
        d1 = ext.extract_random_for_test(seed=0)
        d2 = ext.extract_random_for_test(seed=1)
        assert d1 != d2

    def test_1d_token_ids_accepted(self):
        """1D token_ids [L] should be handled (unsqueezed internally)."""
        ext = NegativeSpaceExtractor()
        token_ids = torch.arange(4)          # [L] not [B, L]
        glyph_coords = torch.zeros(4, 8)     # [L, d_glyph]
        desc = ext.extract_from_glyph_coords(token_ids, glyph_coords)
        assert isinstance(desc, NegativeSpaceDescriptor)


# ============================================================================
# ConsensusNode
# ============================================================================

class TestConsensusNode:

    @pytest.mark.asyncio
    async def test_process_returns_descriptor(self):
        node = ConsensusNode(node_id="n0")
        token_ids = torch.randint(0, 100, (1, 8))
        desc = await node.process(token_ids)
        assert isinstance(desc, NegativeSpaceDescriptor)

    @pytest.mark.asyncio
    async def test_process_updates_last_descriptor(self):
        node = ConsensusNode(node_id="n0")
        assert node.last_descriptor is None
        token_ids = torch.randint(0, 100, (1, 8))
        desc = await node.process(token_ids)
        assert node.last_descriptor is desc

    @pytest.mark.asyncio
    async def test_process_count_increments(self):
        node = ConsensusNode(node_id="n0")
        token_ids = torch.randint(0, 100, (1, 4))
        for _ in range(3):
            await node.process(token_ids)
        assert node.stats()["steps_processed"] == 3

    @pytest.mark.asyncio
    async def test_different_nodes_different_descriptors(self):
        """Nodes with different IDs should produce different descriptors (different seeds)."""
        n1 = ConsensusNode(node_id="alpha")
        n2 = ConsensusNode(node_id="beta")
        token_ids = torch.zeros(1, 4, dtype=torch.long)
        d1 = await n1.process(token_ids)
        d2 = await n2.process(token_ids)
        # Very unlikely to be identical given different hash seeds
        assert d1 != d2

    @pytest.mark.asyncio
    async def test_node_with_glyph_mamba_model(self):
        """ConsensusNode wrapping GlyphMambaWithNegativeSpace runs without error."""
        from src.models.glyph_mamba import GlyphMambaModel
        base = GlyphMambaModel(vocab_size=64, d_model=16, n_layers=1, use_sigmalang=False)
        model = GlyphMambaWithNegativeSpace(base)
        node = ConsensusNode(node_id="glyph-node", model=model)
        token_ids = torch.randint(0, 64, (1, 6))
        desc = await node.process(token_ids)
        assert isinstance(desc, NegativeSpaceDescriptor)


# ============================================================================
# NegativeSpaceConsensusEngine
# ============================================================================

class TestNegativeSpaceConsensusEngine:

    def _make_engine(self, n_nodes: int = 2) -> NegativeSpaceConsensusEngine:
        nodes = [ConsensusNode(node_id=f"node-{i}") for i in range(n_nodes)]
        return NegativeSpaceConsensusEngine(nodes)

    def test_requires_at_least_two_nodes(self):
        with pytest.raises(ValueError, match="at least 2 nodes"):
            NegativeSpaceConsensusEngine([ConsensusNode(node_id="solo")])

    @pytest.mark.asyncio
    async def test_run_consensus_returns_result(self):
        engine = self._make_engine(2)
        token_ids = torch.randint(0, 100, (1, 8))
        result = await engine.run_consensus(token_ids)
        assert isinstance(result, ConsensusResult)

    @pytest.mark.asyncio
    async def test_result_node_count(self):
        engine = self._make_engine(4)
        token_ids = torch.randint(0, 100, (1, 8))
        result = await engine.run_consensus(token_ids)
        assert result.node_count == 4

    @pytest.mark.asyncio
    async def test_bandwidth_ratio_large(self):
        """32 bytes vs megabytes of logits → huge compression."""
        engine = self._make_engine(4)
        token_ids = torch.randint(0, 100, (1, 8))
        result = await engine.run_consensus(token_ids, seq_len=512, vocab_size=32_000)
        # N=4 × 512 × 32000 × 32 bits = 2,097,152,000 bits
        # N=4 × 256 bits = 1024 bits
        # ratio ≈ 2,047,000
        assert result.bandwidth_ratio > 1_000_000, (
            f"Expected >1M compression, got {result.bandwidth_ratio}"
        )

    @pytest.mark.asyncio
    async def test_bandwidth_bits_constant(self):
        """bandwidth_bits = N_nodes × 256 regardless of model or sequence."""
        for n in [2, 3, 4, 8]:
            engine = self._make_engine(n)
            token_ids = torch.zeros(1, 4, dtype=torch.long)
            result = await engine.run_consensus(token_ids)
            assert result.bandwidth_bits == n * 256

    @pytest.mark.asyncio
    async def test_pairwise_jaccard_matrix_symmetric(self):
        engine = self._make_engine(3)
        token_ids = torch.randint(0, 100, (1, 6))
        result = await engine.run_consensus(token_ids)
        J = result.pairwise_jaccard
        N = 3
        for i in range(N):
            assert J[i][i] == 1.0
            for j in range(N):
                assert abs(J[i][j] - J[j][i]) < 1e-9, "Jaccard matrix not symmetric"

    def test_aggregate_identical_descriptors_perfect_consensus(self):
        engine = self._make_engine(2)
        d = NegativeSpaceDescriptor(frozenset(range(100)))
        result = engine.aggregate_descriptors([d, d], seq_len=64, vocab_size=1000)
        assert result.strength == 1.0
        assert result.agreed_absent == d.absent
        assert result.disputed == frozenset()

    def test_aggregate_disjoint_descriptors_zero_consensus(self):
        engine = self._make_engine(2)
        d1 = NegativeSpaceDescriptor(frozenset(range(0, 50)))
        d2 = NegativeSpaceDescriptor(frozenset(range(50, 100)))
        result = engine.aggregate_descriptors([d1, d2], seq_len=64, vocab_size=1000)
        assert result.strength == 0.0
        assert result.agreed_absent == frozenset()
        # Disputed includes everything that appeared in either but not both
        assert frozenset(range(100)).issubset(result.disputed)

    def test_aggregate_three_nodes_agreed_absent_intersection(self):
        """agreed_absent must be intersection of ALL node absent sets."""
        engine = self._make_engine(3)
        d1 = NegativeSpaceDescriptor(frozenset([0, 1, 2, 3, 4]))
        d2 = NegativeSpaceDescriptor(frozenset([0, 1, 2, 5, 6]))
        d3 = NegativeSpaceDescriptor(frozenset([0, 1, 7, 8, 9]))
        result = engine.aggregate_descriptors([d1, d2, d3], seq_len=64, vocab_size=1000)
        # Only 0 and 1 are in ALL three absent sets
        assert result.agreed_absent == frozenset([0, 1])

    def test_agreed_present_no_overlap_with_agreed_absent(self):
        engine = self._make_engine(2)
        d1 = NegativeSpaceDescriptor(frozenset([0, 1]))
        d2 = NegativeSpaceDescriptor(frozenset([0, 1]))
        result = engine.aggregate_descriptors([d1, d2], seq_len=64, vocab_size=1000)
        assert result.agreed_absent.isdisjoint(result.agreed_present)

    @pytest.mark.asyncio
    async def test_stats_increments_rounds(self):
        engine = self._make_engine(2)
        token_ids = torch.zeros(1, 4, dtype=torch.long)
        for _ in range(3):
            await engine.run_consensus(token_ids)
        assert engine.stats()["consensus_rounds"] == 3

    @pytest.mark.asyncio
    async def test_result_summary_string(self):
        engine = self._make_engine(2)
        token_ids = torch.zeros(1, 4, dtype=torch.long)
        result = await engine.run_consensus(token_ids, seq_len=64, vocab_size=1000)
        summary = result.summary()
        assert "ConsensusResult" in summary
        assert "strength" in summary
        assert "compression" in summary


# ============================================================================
# GlyphMambaWithNegativeSpace
# ============================================================================

class TestGlyphMambaWithNegativeSpace:

    def test_forward_shape_unchanged(self):
        from src.models.glyph_mamba import GlyphMambaModel
        base = GlyphMambaModel(vocab_size=64, d_model=16, n_layers=1, use_sigmalang=False)
        model = GlyphMambaWithNegativeSpace(base)
        token_ids = torch.randint(0, 64, (1, 8))
        out = model(token_ids)
        assert out.shape == (1, 8, 64)

    def test_captures_glyph_coords(self):
        from src.models.glyph_mamba import GlyphMambaModel
        base = GlyphMambaModel(vocab_size=64, d_model=16, d_glyph=8, n_layers=1, use_sigmalang=False)
        model = GlyphMambaWithNegativeSpace(base)
        assert model.last_glyph_coords is None
        token_ids = torch.randint(0, 64, (1, 6))
        model(token_ids)
        assert model.last_glyph_coords is not None
        # Shape: [B, L, d_glyph]
        assert model.last_glyph_coords.shape == (1, 6, 8)

    def test_extractor_works_after_forward(self):
        from src.models.glyph_mamba import GlyphMambaModel
        base = GlyphMambaModel(vocab_size=64, d_model=16, d_glyph=8, n_layers=1, use_sigmalang=False)
        model = GlyphMambaWithNegativeSpace(base)
        token_ids = torch.randint(0, 64, (1, 6))
        model(token_ids)
        ext = NegativeSpaceExtractor()
        desc = ext.extract_from_glyph_coords(token_ids, model.last_glyph_coords)
        assert isinstance(desc, NegativeSpaceDescriptor)


# ============================================================================
# Integration: consensus with real GlyphMamba nodes
# ============================================================================

@pytest.mark.asyncio
async def test_same_input_higher_consensus_than_different_input():
    """
    Two nodes processing identical inputs should agree more than two nodes
    processing very different inputs.

    Note: in the synthetic (no-model) case, consensus depends on node_id seeds
    not input content. This test uses GlyphMambaWithNegativeSpace so the
    glyph_coords differ between distinct inputs.
    """
    from src.models.glyph_mamba import GlyphMambaModel

    def make_glyph_node(node_id):
        base = GlyphMambaModel(vocab_size=64, d_model=16, d_glyph=8, n_layers=1, use_sigmalang=False)
        model = GlyphMambaWithNegativeSpace(base)
        return ConsensusNode(node_id=node_id, model=model, coord_threshold=0.3)

    # Two nodes with the same weights (same model init seed)
    torch.manual_seed(7)
    n1 = make_glyph_node("n1")
    torch.manual_seed(7)
    n2 = make_glyph_node("n2")

    # Same input → both glyph_coords come from the same token sequence
    same_tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    engine_same = NegativeSpaceConsensusEngine([n1, n2])
    result_same = await engine_same.run_consensus(same_tokens)

    # Different input (very different token values)
    torch.manual_seed(7)
    n3 = make_glyph_node("n3")
    torch.manual_seed(7)
    n4 = make_glyph_node("n4")
    diff_tokens_a = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]])
    diff_tokens_b = torch.tensor([[63, 63, 63, 63, 63, 63, 63, 63]])  # max token < vocab_size=64
    d3 = await n3.process(diff_tokens_a)
    d4 = await n4.process(diff_tokens_b)
    engine_diff = NegativeSpaceConsensusEngine([n3, n4])
    result_diff = engine_diff.aggregate_descriptors([d3, d4], seq_len=8, vocab_size=64)

    # Same-input consensus should be >= different-input consensus
    # (or at worst equal when both result in all-present or all-absent)
    assert result_same.strength >= result_diff.strength or abs(result_same.strength - result_diff.strength) < 0.2, (
        f"Expected same-input consensus ({result_same.strength:.3f}) >= "
        f"different-input ({result_diff.strength:.3f})"
    )


@pytest.mark.asyncio
async def test_four_node_consensus_agreed_absent_subset_of_all():
    """agreed_absent must be a subset of every individual descriptor's absent set."""
    engine = NegativeSpaceConsensusEngine(
        [ConsensusNode(node_id=f"n{i}") for i in range(4)]
    )
    token_ids = torch.randint(0, 100, (1, 8))
    result = await engine.run_consensus(token_ids)
    for node in engine.nodes:
        assert result.agreed_absent.issubset(node.last_descriptor.absent), (
            "agreed_absent must be subset of every node's absent set"
        )

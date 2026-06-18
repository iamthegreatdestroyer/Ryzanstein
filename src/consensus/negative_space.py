"""
Negative-Space Consensus — Innovation #5
==========================================

Core concept (from Ryot-updates.md, section 6):

    "In multi-node inference, instead of nodes transmitting tokens/latents,
     they transmit **negative-space descriptors** — what they decided NOT to
     compute. Consensus emerges from comparing these absences."

Motivation
----------
Standard distributed inference synchronisation communicates what nodes
*computed*: partial logit tensors, KV-cache shards, or intermediate activations.
For d_model=4096 and vocab_size=32000 this means megabytes per step per node.

Negative-space synchronisation flips the signal: each node summarises which
*glyph primitive regions* contributed negligibly to its forward pass.  The
descriptor is always **32 bytes** (256 bits, one per Σ-glyph primitive) —
independent of d_model, vocab_size, or sequence length.

Protocol
--------
1. Each node runs a forward pass and feeds glyph coordinates through a
   `NegativeSpaceExtractor`.  The extractor marks primitives as "absent" if
   their coordinate magnitude fell below `coord_threshold` — meaning they
   never meaningfully influenced the SSM selectivity at any position.

2. Nodes exchange 32-byte `NegativeSpaceDescriptor` objects.

3. `NegativeSpaceConsensusEngine.aggregate_descriptors()` computes:
   - `agreed_absent`: primitives ALL nodes marked absent → safe to skip
     everywhere on the next forward pass
   - `disputed`: primitives where nodes disagreed → conservative: re-compute
   - `strength`: mean pairwise Jaccard similarity — proxy for confidence

Bandwidth comparison
--------------------
For N=4 nodes, d_model=4096, seq_len=512, vocab=32000:

    Standard (full logits):  N × seq_len × vocab × 4B  = 4 × 512 × 32000 × 4 ≈ 256 MB
    Negative-space:          N × 32B                   = 128 bytes

    Compression ratio ≈ 2,000,000×

Even compared to KV-cache shards (N × seq_len × d_model × 4B ≈ 32 MB),
negative-space is >256,000× smaller.

Key invariants
--------------
- `NegativeSpaceDescriptor.bandwidth_bits` is ALWAYS 256 (32 bytes), regardless
  of how many primitives are absent.
- Jaccard similarity of two descriptors is the consensus strength.
- `merge()` (intersection) is the *conservative* merge: only marks a primitive
  absent if ALL merged descriptors agree it was absent.

Classes
-------
NegativeSpaceDescriptor
    Immutable 256-bit sparse encoding of absent Σ-glyph primitives.
    Wire format: 32-byte bitmask (LSB-first per byte).

NegativeSpaceExtractor
    Derives a NegativeSpaceDescriptor from glyph coordinate embeddings
    produced during a GlyphMambaBlock forward pass.

ConsensusResult
    Aggregation output: strength, agreed_absent, disputed, bandwidth stats.

ConsensusNode
    Wraps a model and extractor.  Processes a token-ID batch and returns
    its NegativeSpaceDescriptor.

NegativeSpaceConsensusEngine
    Orchestrates N ConsensusNodes, exchanges descriptors, and returns a
    ConsensusResult indicating which regions can be safely skipped.
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

N_PRIMITIVES = 256  # Σ-glyph alphabet size — must match GlyphCoordinateEmbedding
_DESCRIPTOR_BYTES = N_PRIMITIVES >> 3  # 32 bytes


# ---------------------------------------------------------------------------
# NegativeSpaceDescriptor
# ---------------------------------------------------------------------------

class NegativeSpaceDescriptor:
    """
    Immutable 256-bit encoding of which Σ-glyph primitives were absent
    (contributed negligibly) during one forward pass.

    Wire format: 32-byte bitmask. Bit (byte_idx*8 + bit_idx) is SET when
    primitive (byte_idx*8 + bit_idx) is ABSENT.

    This is always 256 bits regardless of the number of absent primitives,
    making bandwidth cost constant and independent of model or sequence size.
    """

    __slots__ = ("_absent",)

    def __init__(self, absent_primitives: FrozenSet[int] = frozenset()):
        self._absent: FrozenSet[int] = frozenset(
            int(p) % N_PRIMITIVES for p in absent_primitives
        )

    # ── Wire format ──────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialise to 32-byte bitmask (constant bandwidth cost)."""
        mask = bytearray(_DESCRIPTOR_BYTES)
        for p in self._absent:
            mask[p >> 3] |= 1 << (p & 7)
        return bytes(mask)

    @classmethod
    def from_bytes(cls, data: bytes) -> "NegativeSpaceDescriptor":
        if len(data) != _DESCRIPTOR_BYTES:
            raise ValueError(
                f"NegativeSpaceDescriptor wire format must be {_DESCRIPTOR_BYTES} bytes, "
                f"got {len(data)}"
            )
        absent = set()
        for byte_idx, byte_val in enumerate(data):
            if byte_val:
                for bit_idx in range(8):
                    if byte_val & (1 << bit_idx):
                        absent.add(byte_idx * 8 + bit_idx)
        return cls(frozenset(absent))

    # ── Set operations ───────────────────────────────────────────────────────

    def consensus_with(self, other: "NegativeSpaceDescriptor") -> float:
        """
        Jaccard similarity of absent sets ∈ [0, 1].

        1.0 → perfect agreement (identical absence patterns).
        0.0 → total disagreement (no shared absences).
        Special case: both empty → 1.0 (both nodes fully active — also consensus).
        """
        union = self._absent | other._absent
        if not union:
            return 1.0
        return len(self._absent & other._absent) / len(union)

    def merge(self, other: "NegativeSpaceDescriptor") -> "NegativeSpaceDescriptor":
        """
        Conservative merge: absent only if BOTH descriptors agree.
        (Intersection of absent sets.)

        Use this to accumulate agreed-absent regions safely:
            safe_to_skip = desc_a.merge(desc_b).merge(desc_c)
        """
        return NegativeSpaceDescriptor(self._absent & other._absent)

    def union(self, other: "NegativeSpaceDescriptor") -> "NegativeSpaceDescriptor":
        """
        Liberal merge: absent if EITHER descriptor marks it absent.
        (Union — use for finding any-absent regions for diagnostic purposes.)
        """
        return NegativeSpaceDescriptor(self._absent | other._absent)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def absent(self) -> FrozenSet[int]:
        return self._absent

    @property
    def absent_count(self) -> int:
        return len(self._absent)

    @property
    def sparsity(self) -> float:
        """Fraction of primitives marked absent ∈ [0, 1]."""
        return len(self._absent) / N_PRIMITIVES

    @property
    def bandwidth_bits(self) -> int:
        """Always 256 — constant cost regardless of sparsity."""
        return N_PRIMITIVES

    def __repr__(self) -> str:
        return (
            f"NegativeSpaceDescriptor("
            f"absent={self.absent_count}/{N_PRIMITIVES}, "
            f"sparsity={self.sparsity:.1%})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NegativeSpaceDescriptor):
            return NotImplemented
        return self._absent == other._absent

    def __hash__(self) -> int:
        return hash(self._absent)


# ---------------------------------------------------------------------------
# NegativeSpaceExtractor
# ---------------------------------------------------------------------------

class NegativeSpaceExtractor:
    """
    Derives a NegativeSpaceDescriptor from glyph coordinate embeddings.

    A glyph primitive is "absent" at a given inference step if every token
    position that mapped to it had a glyph coordinate embedding whose L2 norm
    fell below `coord_threshold`.  A low-norm embedding means the glyph
    dimension contributed negligibly to the SSM's B, C, and Δ projections.

    Alternatively, if raw SSM Δ values are available, `extract_from_delta()`
    uses them directly — Δ < `delta_threshold` per-position means the token
    barely updated the SSM hidden state.

    Both methods fall back to `token_id % 256` primitive bucketing when
    sigmalang is unavailable.
    """

    def __init__(
        self,
        coord_threshold: float = 0.10,
        delta_threshold: float = 0.01,
    ):
        self.coord_threshold = coord_threshold
        self.delta_threshold = delta_threshold
        self._mapper = None
        self._try_load_mapper()

    def _try_load_mapper(self) -> None:
        try:
            import sys
            from pathlib import Path
            _sl_root = Path(__file__).parent.parent.parent.parent / "sigmalang"
            if _sl_root.exists():
                sys.path.insert(0, str(_sl_root))
            from src.recycler.glyph_kv_cache import TokenGlyphMapper
            self._mapper = TokenGlyphMapper()
        except Exception:
            pass

    def _token_to_primitive(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._mapper is not None:
            flat = token_ids.reshape(-1).tolist()
            prims = [self._mapper._map_one(t).primitive_id for t in flat]
            return torch.tensor(prims, dtype=torch.long, device=token_ids.device).reshape(token_ids.shape)
        return token_ids % N_PRIMITIVES

    def extract_from_glyph_coords(
        self,
        token_ids: torch.Tensor,   # [B, L] or [L]
        glyph_coords: torch.Tensor, # [B, L, d_glyph] or [L, d_glyph]
    ) -> NegativeSpaceDescriptor:
        """
        Absent primitive = one where ALL token positions that map to it had
        glyph coordinate norm below `coord_threshold`.

        Primitives that never appeared in this batch are neither marked absent
        nor present — they are simply not considered.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            glyph_coords = glyph_coords.unsqueeze(0) if glyph_coords.dim() == 2 else glyph_coords

        coord_norms = glyph_coords.norm(dim=-1)    # [B, L]
        primitive_ids = self._token_to_primitive(token_ids)  # [B, L]

        absent: set = set()
        for prim_id in range(N_PRIMITIVES):
            mask = (primitive_ids == prim_id)      # [B, L] bool
            if not mask.any():
                continue
            # Absent if ALL appearances have low-norm glyph coords
            if (coord_norms[mask] < self.coord_threshold).all():
                absent.add(prim_id)

        return NegativeSpaceDescriptor(frozenset(absent))

    def extract_from_delta(
        self,
        token_ids: torch.Tensor,   # [B, L] or [L]
        delta: torch.Tensor,       # [B, L, d_model] — SSM step sizes (post-softplus)
    ) -> NegativeSpaceDescriptor:
        """
        Absent primitive = one where ALL positions mapping to it had
        mean(Δ_t) < delta_threshold — the SSM step was negligibly small.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            delta = delta.unsqueeze(0) if delta.dim() == 2 else delta

        delta_mean = delta.mean(dim=-1)            # [B, L]
        primitive_ids = self._token_to_primitive(token_ids)

        absent: set = set()
        for prim_id in range(N_PRIMITIVES):
            mask = (primitive_ids == prim_id)
            if not mask.any():
                continue
            if (delta_mean[mask] < self.delta_threshold).all():
                absent.add(prim_id)

        return NegativeSpaceDescriptor(frozenset(absent))

    def extract_random_for_test(
        self,
        seed: int = 0,
        absent_fraction: float = 0.6,
    ) -> NegativeSpaceDescriptor:
        """
        Deterministic synthetic descriptor for testing without a real model.
        `absent_fraction` of primitives are marked absent.
        """
        import random
        rng = random.Random(seed)
        absent = frozenset(
            p for p in range(N_PRIMITIVES)
            if rng.random() < absent_fraction
        )
        return NegativeSpaceDescriptor(absent)


# ---------------------------------------------------------------------------
# ConsensusResult
# ---------------------------------------------------------------------------

@dataclass
class ConsensusResult:
    """
    Aggregated output of a multi-node consensus round.

    Attributes
    ----------
    strength
        Mean pairwise Jaccard similarity across all node-pair descriptors.
        1.0 = all nodes agreed perfectly; 0.0 = total disagreement.
    agreed_absent
        Primitives marked absent by ALL nodes — safe to skip on next step.
    agreed_present
        Primitives that were NOT absent in any node — always computed.
    disputed
        Primitives where at least one node differed — re-compute conservatively.
    node_count
        Number of nodes that participated.
    bandwidth_bits
        Total bits transmitted for the consensus exchange (N × 256).
    baseline_bits
        Bits that would be needed to share full logit vectors
        (N × seq_len × vocab_size × 32 bits).
    bandwidth_ratio
        baseline_bits / bandwidth_bits — the compression factor.
    pairwise_jaccard
        Full pairwise Jaccard matrix for diagnostics.
    """
    strength: float
    agreed_absent: FrozenSet[int]
    agreed_present: FrozenSet[int]
    disputed: FrozenSet[int]
    node_count: int
    bandwidth_bits: int
    baseline_bits: int
    bandwidth_ratio: float
    pairwise_jaccard: List[List[float]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ConsensusResult("
            f"strength={self.strength:.3f}, "
            f"agreed_absent={len(self.agreed_absent)}, "
            f"disputed={len(self.disputed)}, "
            f"bandwidth={self.bandwidth_bits}b vs {self.baseline_bits}b "
            f"[{self.bandwidth_ratio:.0f}x compression])"
        )


# ---------------------------------------------------------------------------
# ConsensusNode
# ---------------------------------------------------------------------------

class ConsensusNode:
    """
    A single inference node in the multi-node consensus mesh.

    Wraps a model (or any callable) and a NegativeSpaceExtractor.
    On `process()`, runs the model and derives its NegativeSpaceDescriptor.

    The model interface is:
        output: Tensor = model(token_ids)      [B, L, d_model] or [B, L, vocab]
        glyph_coords: Tensor = model.last_glyph_coords  [B, L, d_glyph]  (optional)

    If `model` does not expose `last_glyph_coords`, the extractor falls back to
    random synthetic descriptors (useful for testing the consensus protocol
    independently of any specific model).
    """

    def __init__(
        self,
        node_id: str,
        model: Optional[nn.Module] = None,
        extractor: Optional[NegativeSpaceExtractor] = None,
        coord_threshold: float = 0.10,
    ):
        self.node_id = node_id
        self.model = model
        self.extractor = extractor or NegativeSpaceExtractor(coord_threshold=coord_threshold)
        self._last_descriptor: Optional[NegativeSpaceDescriptor] = None
        self._process_count = 0

    async def process(
        self,
        token_ids: torch.Tensor,
    ) -> NegativeSpaceDescriptor:
        """
        Run one inference step and return this node's NegativeSpaceDescriptor.

        If the model exposes `last_glyph_coords`, uses them for extraction.
        Otherwise uses a deterministic synthetic descriptor seeded by
        (node_id_hash, step_count) — consistent within a node, diverse across nodes.
        """
        self._process_count += 1

        if self.model is not None and hasattr(self.model, "last_glyph_coords"):
            with torch.no_grad():
                _ = self.model(token_ids)
            glyph_coords = self.model.last_glyph_coords
            desc = self.extractor.extract_from_glyph_coords(token_ids, glyph_coords)
        elif self.model is not None:
            # Run model but extract from token_ids using synthetic absent computation
            with torch.no_grad():
                _ = self.model(token_ids)
            # Derive synthetic descriptor deterministically from node + step
            seed = hash(self.node_id) ^ self._process_count
            desc = self.extractor.extract_random_for_test(seed=seed & 0xFFFFFFFF)
        else:
            seed = hash(self.node_id) ^ self._process_count
            desc = self.extractor.extract_random_for_test(seed=seed & 0xFFFFFFFF)

        self._last_descriptor = desc
        return desc

    @property
    def last_descriptor(self) -> Optional[NegativeSpaceDescriptor]:
        return self._last_descriptor

    def stats(self) -> Dict:
        return {
            "node_id": self.node_id,
            "steps_processed": self._process_count,
            "last_sparsity": (
                self._last_descriptor.sparsity if self._last_descriptor else None
            ),
        }


# ---------------------------------------------------------------------------
# GlyphMambaWithNegativeSpace
# ---------------------------------------------------------------------------

class GlyphMambaWithNegativeSpace(nn.Module):
    """
    Thin wrapper around GlyphMambaModel that captures glyph coordinates from the
    first block's embedding during forward — exposing `last_glyph_coords` for the
    NegativeSpaceExtractor.

    Usage:
        model = GlyphMambaWithNegativeSpace(base_model)
        output = model(token_ids)
        desc = extractor.extract_from_glyph_coords(token_ids, model.last_glyph_coords)
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.last_glyph_coords: Optional[torch.Tensor] = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Hook into first block's glyph embedding to capture coords
        _coords: List[torch.Tensor] = []

        def _hook(module, inp, out):
            _coords.append(out.detach())

        handle = None
        if hasattr(self.base, "blocks") and len(self.base.blocks) > 0:
            first_block = self.base.blocks[0]
            if hasattr(first_block, "glyph_embed"):
                handle = first_block.glyph_embed.register_forward_hook(_hook)

        out = self.base(token_ids)

        if handle is not None:
            handle.remove()
        if _coords:
            self.last_glyph_coords = _coords[0]

        return out


# ---------------------------------------------------------------------------
# NegativeSpaceConsensusEngine
# ---------------------------------------------------------------------------

class NegativeSpaceConsensusEngine:
    """
    Orchestrates N ConsensusNodes and aggregates their NegativeSpaceDescriptors.

    The engine:
    1. Fans out the same token batch to all N nodes in parallel.
    2. Each node returns its 32-byte NegativeSpaceDescriptor.
    3. Aggregates: computes pairwise Jaccard, finds agreed/disputed regions.
    4. Returns a ConsensusResult with bandwidth statistics.

    The key output is `result.agreed_absent` — the set of glyph primitives
    that ALL nodes consider irrelevant.  On the next inference step, a
    primitive in `agreed_absent` can be safely skipped by all nodes without
    exchanging data, since no node needs it.

    Parameters
    ----------
    nodes
        List of ConsensusNode instances (simulating separate inference servers).
    seq_len_hint
        Used for baseline bandwidth calculation. Defaults to 512.
    vocab_size_hint
        Used for baseline bandwidth calculation. Defaults to 32000.
    """

    def __init__(
        self,
        nodes: List[ConsensusNode],
        seq_len_hint: int = 512,
        vocab_size_hint: int = 32_000,
    ):
        if len(nodes) < 2:
            raise ValueError("NegativeSpaceConsensusEngine requires at least 2 nodes")
        self.nodes = nodes
        self.seq_len_hint = seq_len_hint
        self.vocab_size_hint = vocab_size_hint
        self._rounds: int = 0

    async def run_consensus(
        self,
        token_ids: torch.Tensor,
        seq_len: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ) -> ConsensusResult:
        """
        Fan out token_ids to all nodes, collect descriptors, aggregate.

        All nodes run concurrently via asyncio.gather.
        Returns ConsensusResult with bandwidth stats and agreed regions.
        """
        self._rounds += 1
        seq_len   = seq_len   or self.seq_len_hint
        vocab_size = vocab_size or self.vocab_size_hint

        # Parallel dispatch
        descriptors: List[NegativeSpaceDescriptor] = await asyncio.gather(
            *[node.process(token_ids) for node in self.nodes]
        )

        return self.aggregate_descriptors(
            descriptors,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )

    def aggregate_descriptors(
        self,
        descriptors: Sequence[NegativeSpaceDescriptor],
        seq_len: int = 512,
        vocab_size: int = 32_000,
    ) -> ConsensusResult:
        """
        Pure aggregation — no I/O, fully synchronous.  Exposed separately for
        testing and for cases where callers collect descriptors via their own
        transport (gRPC, etc.).
        """
        N = len(descriptors)
        if N == 0:
            raise ValueError("At least one descriptor required")

        # ── Pairwise Jaccard ────────────────────────────────────────────────
        pairwise = [[0.0] * N for _ in range(N)]
        jaccard_sum = 0.0
        pair_count  = 0
        for i in range(N):
            pairwise[i][i] = 1.0
            for j in range(i + 1, N):
                j_val = descriptors[i].consensus_with(descriptors[j])
                pairwise[i][j] = j_val
                pairwise[j][i] = j_val
                jaccard_sum += j_val
                pair_count  += 1

        strength = jaccard_sum / pair_count if pair_count else 1.0

        # ── Agreed absent (intersection of ALL absent sets) ─────────────────
        agreed_absent = descriptors[0].absent
        for d in descriptors[1:]:
            agreed_absent = agreed_absent & d.absent
        agreed_absent = frozenset(agreed_absent)

        # ── Agreed present (none marked absent in any node) ─────────────────
        any_absent = set()
        for d in descriptors:
            any_absent |= d.absent
        agreed_present = frozenset(range(N_PRIMITIVES)) - any_absent

        # ── Disputed (at least one node disagrees) ───────────────────────────
        disputed: set = set()
        for i in range(N):
            for j in range(i + 1, N):
                sym_diff = descriptors[i].absent ^ descriptors[j].absent
                disputed |= sym_diff
        disputed_frozen = frozenset(disputed)

        # ── Bandwidth stats ──────────────────────────────────────────────────
        # Negative-space: each descriptor is always 256 bits (32 bytes)
        bandwidth_bits = N * N_PRIMITIVES
        # Baseline: full logit tensors per node per step (float32)
        baseline_bits  = N * seq_len * vocab_size * 32
        bandwidth_ratio = baseline_bits / bandwidth_bits if bandwidth_bits else 0.0

        return ConsensusResult(
            strength=round(strength, 6),
            agreed_absent=agreed_absent,
            agreed_present=agreed_present,
            disputed=disputed_frozen,
            node_count=N,
            bandwidth_bits=bandwidth_bits,
            baseline_bits=baseline_bits,
            bandwidth_ratio=round(bandwidth_ratio, 1),
            pairwise_jaccard=pairwise,
        )

    def stats(self) -> Dict:
        return {
            "node_count": len(self.nodes),
            "consensus_rounds": self._rounds,
            "nodes": [n.stats() for n in self.nodes],
        }

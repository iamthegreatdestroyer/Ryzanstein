"""
Recursive Fractality in Quantization (Probabilistic Lattice Fractal) — Innovation #2
======================================================================================

Core concept (from Ryot-updates.md, section 2):

    "Fractal quantization: Each bit level represents a different scale of
     semantic fidelity. Querying the glyph manifold at bit-level N gives you
     a lossy-but-fast approximation; bit-level N+k gives you higher fidelity.
     Use QHSS structure to encode uncertainty not as separate metadata, but
     as lattice topology. The quantization IS the uncertainty estimate.
     This enables adaptive inference: the system asks 'how much fidelity do
     I need for this query?' and reads only that many bits — sub-linear
     both computationally and informationally."

Also incorporates Innovation #10 (Quantum-Friendly Design):

    "Design the glyph algebra so it maps cleanly onto quantum operations
     (tensor products, interference patterns). This doesn't require quantum
     hardware now, but makes a future quantum co-processor plug-and-play
     without architectural rewrites. The probabilistic lattice quantization
     + negative space computations are already quantum-friendly in philosophy."

Architecture
------------

ProbabilisticLattice
    A multi-scale lattice encoding a value at B bit levels. Each level stores
    a coarser representation; reading level k gives fidelity 2^(-k).

    The lattice IS the uncertainty estimate — the difference between level k
    and level k+1 encodes the quantization error at that scale.
    No separate uncertainty metadata needed.

    Layout (for a scalar value v ∈ [0, 1] at B bit levels):
        level 0:  1-bit approximation  (sign / coarser half)
        level 1:  2-bit approximation
        level k:  2^k-step approximation
        level B:  full 2^B-step resolution

    Quantum-friendly: each level corresponds to one qubit register.
    Reading levels 0..k is equivalent to projecting the quantum state onto
    the k-qubit subspace — no architectural change needed for quantum execution.

FractalQuantizer
    Quantizes glyph embedding vectors (float32[]) to ProbabilisticLattice
    representations, and dequantizes back at any requested fidelity level.

    Key operation: query(v, bits=k) — returns the k-bit approximation in O(k),
    not O(B). Reading fewer bits = faster, more uncertain; more bits = slower, precise.

    Cross-project synergy: mirrors NLCI hierarchical uncertainty structure and
    Negative_Space_Imaging adaptive resolution.

AdaptiveInferenceQuery
    The runtime decision function: given a query context (glyph centroid),
    determines how many bits of quantization fidelity are needed.

    Strategy: start at bits=1 (coarsest), refine until the approximation
    agrees with the previous level within tolerance. This is the adaptive
    part — easy queries need 2-3 bits; hard queries need full B bits.
"""

import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Default bit depth for the fractal lattice
DEFAULT_BIT_LEVELS = 8

# Quantum-friendly: 8 levels = 8 qubits = 256-step resolution
# Matches N_PRIMITIVES (256) — intentional structural alignment
N_PRIMITIVES = 256


# ---------------------------------------------------------------------------
# QuantizationLevel
# ---------------------------------------------------------------------------

@dataclass
class QuantizationLevel:
    """
    One level in the fractal quantization hierarchy.

    bits:       number of bits at this level
    resolution: 2^bits steps in [0, 1]
    value:      the approximated value at this resolution
    error:      absolute quantization error at this level
    uncertainty: error as fraction of value range (= 1.0 / resolution)
    """
    bits:        int
    resolution:  int
    value:       float
    error:       float
    uncertainty: float

    @property
    def fidelity(self) -> float:
        """1 - uncertainty, in [0, 1]. Higher bits → higher fidelity."""
        return 1.0 - self.uncertainty


# ---------------------------------------------------------------------------
# ProbabilisticLattice
# ---------------------------------------------------------------------------

class ProbabilisticLattice:
    """
    Multi-scale lattice encoding of a scalar value ∈ [0, 1].

    Each level k stores the 2^k-step quantized approximation of the value.
    The lattice IS the uncertainty: level k has uncertainty 1/2^k, no need
    for separate error bars.

    Storage: B integers (one per level), each in [0, 2^k - 1].
    Memory: B × ceil(log2(max_level)) bits ≈ B × B bits total.

    For B=8: 8 × 8 = 64 bits = 8 bytes per scalar — identical to float64 but
    with built-in multi-resolution access.

    Quantum interpretation:
        - Level k = register of k qubits measuring the value
        - Reading level k = projecting onto the k-qubit measurement basis
        - Full quantum state = superposition of all levels simultaneously
        - Classical simulation is exact; quantum execution adds parallelism
    """

    __slots__ = ("_levels", "_b")

    def __init__(self, levels: List[int], b: int = DEFAULT_BIT_LEVELS):
        if len(levels) != b:
            raise ValueError(f"Expected {b} levels, got {len(levels)}")
        self._levels = list(levels)
        self._b = b

    @classmethod
    def encode(cls, value: float, b: int = DEFAULT_BIT_LEVELS) -> "ProbabilisticLattice":
        """
        Encode a scalar value ∈ [0, 1] into a B-level fractal lattice.

        Each level k stores floor(value × 2^k), so:
            level 0: {0, 1}
            level 1: {0, 1, 2, 3}
            level k: {0, ..., 2^k - 1}
        """
        v = max(0.0, min(1.0, float(value)))
        levels = []
        for k in range(1, b + 1):
            steps = 1 << k
            levels.append(int(v * steps))
        return cls(levels, b)

    def query(self, bits: int) -> QuantizationLevel:
        """
        Query the value at the given bit level.

        O(1) — reads one precomputed level directly.
        No recomputation from full-resolution data.

        Args:
            bits: fidelity level in [1, B]

        Returns:
            QuantizationLevel with value, error, and uncertainty at this resolution.
        """
        bits = max(1, min(bits, self._b))
        idx = bits - 1
        steps = 1 << bits
        quantised = self._levels[idx]
        v = quantised / steps
        uncertainty = 1.0 / steps          # maximum quantisation error
        exact = self._levels[-1] / (1 << self._b)  # full-resolution estimate
        error = abs(v - exact)
        return QuantizationLevel(
            bits=bits, resolution=steps,
            value=v, error=error, uncertainty=uncertainty,
        )

    def query_range(self, min_bits: int, max_bits: int) -> List[QuantizationLevel]:
        """Return all levels from min_bits to max_bits (inclusive)."""
        return [self.query(k) for k in range(min_bits, max_bits + 1)]

    def full_value(self) -> float:
        """Full-resolution reconstruction (bits = B)."""
        return self.query(self._b).value

    def uncertainty_at(self, bits: int) -> float:
        """Uncertainty (= 1/resolution) at a given bit level."""
        return 1.0 / (1 << max(1, min(bits, self._b)))

    def level_delta(self, bits: int) -> float:
        """
        Information gained by going from bits-1 to bits.

        Positive = higher level refined the estimate.
        Large delta = this query needed more bits to stabilise.

        Quantum analogy: measuring one more qubit (projecting onto a smaller
        subspace) reduces uncertainty by exactly this delta.
        """
        if bits <= 1:
            return self.query(1).value
        prev = self.query(bits - 1).value
        curr = self.query(bits).value
        return abs(curr - prev)

    def to_bytes(self) -> bytes:
        """Compact encoding: B bytes (one uint8 per level, values up to 2^8 = 256)."""
        return bytes(min(v, 255) for v in self._levels)

    @classmethod
    def from_bytes(cls, data: bytes, b: int = DEFAULT_BIT_LEVELS) -> "ProbabilisticLattice":
        if len(data) < b:
            raise ValueError(f"Need at least {b} bytes")
        return cls(list(data[:b]), b)

    def __repr__(self) -> str:
        v = self.full_value()
        return f"ProbabilisticLattice(value~{v:.4f}, bits={self._b})"


# ---------------------------------------------------------------------------
# FractalQuantizer
# ---------------------------------------------------------------------------

class FractalQuantizer:
    """
    Quantizes float32 tensors to ProbabilisticLattice representations and
    dequantizes back at any requested fidelity level.

    Designed for glyph embedding vectors (shape [*, d_glyph]) but works on
    any float tensor.

    The per-element range [min, max] is normalised to [0, 1] before encoding.
    Range parameters are stored for dequantisation.
    """

    def __init__(self, bit_levels: int = DEFAULT_BIT_LEVELS):
        self.bit_levels = bit_levels

    def quantize(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[List[ProbabilisticLattice]], torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to a list of ProbabilisticLattice objects.

        Args:
            x: float tensor of any shape

        Returns:
            (lattices, x_min, x_max)
                lattices: nested list mirroring x.shape, each element a ProbabilisticLattice
                x_min, x_max: per-tensor range for dequantisation
        """
        flat = x.reshape(-1).float()
        x_min = flat.min()
        x_max = flat.max()
        rng = x_max - x_min
        if rng < 1e-8:
            rng = torch.tensor(1.0)
        normalised = ((flat - x_min) / rng).tolist()
        lattices = [
            ProbabilisticLattice.encode(v, self.bit_levels)
            for v in normalised
        ]
        return lattices, x_min, x_max

    def dequantize(
        self,
        lattices: List[ProbabilisticLattice],
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        bits: int,
        shape: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Dequantize at a given bit level — O(N) not O(N × B).

        Args:
            lattices:  list of ProbabilisticLattice (one per element)
            x_min, x_max: range from quantize()
            bits:      fidelity level 1..B
            shape:     original tensor shape (optional)

        Returns:
            float32 tensor at requested fidelity
        """
        values = [lat.query(bits).value for lat in lattices]
        t = torch.tensor(values, dtype=torch.float32)
        rng = x_max - x_min
        t = t * rng + x_min
        if shape is not None:
            t = t.reshape(shape)
        return t

    def query_adaptive(
        self,
        lattices: List[ProbabilisticLattice],
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        shape: Optional[Tuple],
        tolerance: float = 0.01,
        min_bits: int = 2,
        max_bits: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Adaptive query: start coarse, refine until consecutive levels agree
        within tolerance.

        This is the key sub-linear property: easy queries stop at 2-3 bits;
        hard queries continue to full B bits.

        Returns:
            (dequantised_tensor, bits_used, mean_delta)
        """
        max_b = max_bits or self.bit_levels
        prev = self.dequantize(lattices, x_min, x_max, min_bits, shape)
        for k in range(min_bits + 1, max_b + 1):
            curr = self.dequantize(lattices, x_min, x_max, k, shape)
            delta = (curr - prev).abs().mean().item()
            if delta < tolerance:
                return curr, k, delta
            prev = curr
        return prev, max_b, 0.0

    def bits_used_histogram(
        self,
        sample_lattices: List[List[ProbabilisticLattice]],
        x_mins: List[torch.Tensor],
        x_maxs: List[torch.Tensor],
        tolerance: float = 0.01,
    ) -> Dict[int, int]:
        """
        Count how many bits each sample needed for adaptive convergence.
        Useful for tuning tolerance and understanding workload distribution.
        """
        hist: Dict[int, int] = {}
        for lats, xm, xM in zip(sample_lattices, x_mins, x_maxs):
            _, bits, _ = self.query_adaptive(lats, xm, xM, shape=None, tolerance=tolerance)
            hist[bits] = hist.get(bits, 0) + 1
        return hist


# ---------------------------------------------------------------------------
# AdaptiveInferenceQuery
# ---------------------------------------------------------------------------

class AdaptiveInferenceQuery:
    """
    Runtime decision function: maps a glyph centroid context to an appropriate
    quantization bit level.

    Strategy (from doc): "the system asks 'how much fidelity do I need for
    this query?' and reads only that many bits."

    Implementation: linear regression over glyph centroid sparsity and
    consensus strength → estimated required bits. No model weights needed —
    uses the glyph coordinate statistics.

    Heuristic rules (tuned from observation, not training):
        - High sparsity (few primitives active) → low fidelity needed (fewer bits)
          rationale: sparse queries have high semantic certainty
        - High centroid entropy → more bits needed (diverse glyph mix)
        - Consensus strength (from NegativeSpaceConsensusEngine) → fewer bits
          needed when nodes agree (high confidence in glyph neighbourhood)

    For regulated-domain use: combine with GlyphFingerprintChain to prove
    which fidelity level was used for each inference step.
    """

    def __init__(
        self,
        min_bits:          int   = 2,
        max_bits:          int   = DEFAULT_BIT_LEVELS,
        sparsity_weight:   float = -4.0,  # negative: sparse → fewer bits
        entropy_weight:    float = +3.0,  # positive: high entropy → more bits
        consensus_weight:  float = -2.0,  # negative: high consensus → fewer bits
    ):
        self.min_bits         = min_bits
        self.max_bits         = max_bits
        self.sparsity_weight  = sparsity_weight
        self.entropy_weight   = entropy_weight
        self.consensus_weight = consensus_weight
        self._query_history: List[Tuple[float, int]] = []  # (sparsity, bits_used)

    def estimate_bits(
        self,
        centroid: List[float],
        consensus_strength: float = 0.0,
    ) -> int:
        """
        Estimate required bit level from glyph centroid statistics.

        Args:
            centroid:           256-dim histogram (from GlyphNativeStream.to_centroid())
            consensus_strength: Jaccard mean from NegativeSpaceConsensusEngine (0..1)

        Returns:
            Estimated bit level in [min_bits, max_bits]
        """
        active = sum(1 for c in centroid if c > 0)
        # Complement fraction: 1 = fully sparse (few active), 0 = fully dense
        sparsity = 1.0 - active / N_PRIMITIVES

        # Shannon entropy of centroid distribution
        entropy = 0.0
        for c in centroid:
            if c > 1e-9:
                entropy -= c * math.log2(c)
        # Normalise to [0, 1] (max entropy = log2(256) ≈ 8.0)
        entropy_norm = min(1.0, entropy / 8.0)

        raw = (
            self.max_bits / 2.0
            + self.sparsity_weight  * sparsity
            + self.entropy_weight   * entropy_norm
            + self.consensus_weight * consensus_strength
        )
        bits = int(round(max(self.min_bits, min(self.max_bits, raw))))
        self._query_history.append((sparsity, bits))
        return bits

    def stats(self) -> Dict:
        if not self._query_history:
            return {"queries": 0}
        bits_list = [b for _, b in self._query_history]
        return {
            "queries":    len(bits_list),
            "mean_bits":  round(sum(bits_list) / len(bits_list), 2),
            "min_bits":   min(bits_list),
            "max_bits":   max(bits_list),
            "sub_max_pct": round(
                100.0 * sum(1 for b in bits_list if b < self.max_bits) / len(bits_list), 1
            ),
        }

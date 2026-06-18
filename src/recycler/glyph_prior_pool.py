"""
Bidirectional Token Recycling — Glyph Prior Pool
=================================================

Innovation #3 from Ryot-updates.md:

    "When you retrieve a cached context glyph, don't just inject it as KV
    values. Use it to seed the next glyph generation task itself. The recycled
    glyph becomes a 'prior' that shapes which new glyphs are generated."

The GlyphPriorPool maintains a weighted distribution over glyph primitive IDs.
Each inference call leaves a "glyph residue" — an increment to the primitives it
touched. Future generation steps receive these accumulated residues as a logit
bias, softly steering output toward glyphs that appeared frequently in recent
context. The effect is emergent coherence without explicit message passing.

Key properties:
  - Exponential decay so stale residues fade (no perpetual bias)
  - Normalized bias: capped at `prior_strength` so the model's own logits dominate
  - Thread-safe-enough for single-process async use (pure Python dict ops)
  - Works gracefully when sigmalang is absent (bias stays at zero)
  - Bias is sparse: only N_active_primitives tokens (≤256) get non-zero bias

Usage
-----
    pool = GlyphPriorPool(vocab_size=32_000, prior_strength=0.15)
    pool.accumulate([1, 2, 3, 256, 384])      # leave residue after generation
    pool.decay()                               # call periodically (per batch)
    prior = pool.prior_tensor(vocab_size=32_000)  # add to logits
    logits = model(tokens) + prior.to(logits.device)
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING

try:
    from sigmalang.core.primitives import GlyphType
    _SIGMALANG_AVAILABLE = True
except ImportError:
    _SIGMALANG_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class GlyphPriorPool:
    """
    Maintains a per-primitive-ID weight table built from past token sequences.

    Two public entry points:
      accumulate(token_ids)    — add residue (call after each generation)
      prior_logits(vocab_size) — read bias (call before generation)

    The prior is injected additively into model logits, scaled to
    `prior_strength` so it nudges rather than dominates.
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        decay_factor: float = 0.95,
        prior_strength: float = 0.15,
    ):
        """
        Args:
            vocab_size:     Token vocabulary size (for prior_tensor shape).
            decay_factor:   Multiplier applied each decay() call (0.9–0.99 typical).
            prior_strength: Maximum logit bias magnitude. 0.15 is a gentle nudge;
                            1.0 would strongly prefer recycled glyphs.
        """
        # primitive_id (0-255) → accumulated weight
        self._weights: Dict[int, float] = defaultdict(float)
        self.vocab_size = vocab_size
        self.decay_factor = decay_factor
        self.prior_strength = prior_strength

        # Internal state
        self._total_accumulations: int = 0
        self._total_decays: int = 0

        # Lazy mapper — only imported when sigmalang is available
        self._mapper = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def accumulate(self, token_ids: List[int], weight: float = 1.0) -> None:
        """
        Add glyph residue from a completed token sequence.

        Each glyph in the sequence increments the weight of its primitive_id
        by `weight / sequence_length` (normalised so short/long sequences
        contribute equally by default).

        Args:
            token_ids: Token sequence that was just generated / processed.
            weight:    Contribution scale — use <1.0 for lower-priority residues.
        """
        if not _SIGMALANG_AVAILABLE or not token_ids:
            return
        try:
            mapper = self._get_mapper()
            glyphs = mapper.tokens_to_glyphs(token_ids)
            per_glyph = weight / max(len(glyphs), 1)
            for g in glyphs:
                self._weights[g.primitive_id] += per_glyph
            self._total_accumulations += 1
        except Exception:
            pass  # never break inference for prior accounting

    def decay(self) -> None:
        """
        Apply exponential time-decay to all residues.

        Call once per batch (or on a timer) to prevent stale residues from
        permanently biasing generation. After ~log(1e6)/log(1/0.95) ≈ 275 calls,
        a weight that started at 1.0 drops below 1e-6 and is pruned.
        """
        dead: List[int] = []
        for pid, w in self._weights.items():
            new_w = w * self.decay_factor
            if new_w < 1e-7:
                dead.append(pid)
            else:
                self._weights[pid] = new_w
        for pid in dead:
            del self._weights[pid]
        self._total_decays += 1

    def prior_logits(self, vocab_size: Optional[int] = None) -> List[float]:
        """
        Compute a logit-bias vector aligned to the token vocabulary.

        For each token ID, its primitive_id is looked up in the weight table.
        The bias is max-normalised to `prior_strength` to ensure it never
        dominates the model's own predictions.

        Returns a list of `vocab_size` floats (zeros when sigmalang absent).
        Time: O(vocab_size) — acceptable since model forward is O(vocab_size) too.
        """
        n = vocab_size or self.vocab_size
        bias = [0.0] * n
        if not _SIGMALANG_AVAILABLE or not self._weights:
            return bias

        try:
            mapper = self._get_mapper()
            max_w = max(self._weights.values())
            if max_w == 0.0:
                return bias
            scale = self.prior_strength / max_w

            for tok_id in range(n):
                g = mapper._map_one(tok_id)
                w = self._weights.get(g.primitive_id, 0.0)
                if w > 0.0:
                    bias[tok_id] = w * scale
        except Exception:
            pass

        return bias

    def prior_tensor(self, vocab_size: Optional[int] = None):
        """Return prior as a torch.FloatTensor [vocab_size]. Requires torch."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch not available")
        import torch
        return torch.tensor(self.prior_logits(vocab_size), dtype=torch.float32)

    def reset(self) -> None:
        """Clear all accumulated residues."""
        self._weights.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        max_w = max(self._weights.values(), default=0.0)
        return {
            "active_primitives": len(self._weights),
            "total_accumulations": self._total_accumulations,
            "total_decays": self._total_decays,
            "max_weight": round(max_w, 6),
            "prior_strength": self.prior_strength,
            "decay_factor": self.decay_factor,
            "sigmalang_available": _SIGMALANG_AVAILABLE,
        }

    def top_primitives(self, n: int = 10) -> List[Dict]:
        """Return the n highest-weight primitive IDs (for debugging / monitoring)."""
        sorted_items = sorted(self._weights.items(), key=lambda x: -x[1])
        return [
            {"primitive_id": pid, "weight": round(w, 6)}
            for pid, w in sorted_items[:n]
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_mapper(self):
        if self._mapper is None:
            from .glyph_kv_cache import TokenGlyphMapper
            self._mapper = TokenGlyphMapper()
        return self._mapper

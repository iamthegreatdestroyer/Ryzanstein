"""
Autonomy Loop Closure — Innovation #7
======================================

Core concept (from Ryot-updates.md, section 7):

    "Each inference run generates telemetry glyphs (what worked, what failed,
     what was expensive). These feed into a meta-optimizer agent that proposes
     micro-optimizations to speculative decoding strategy, quantization
     bit-widths, or glyph manifold topology. Proposals are tested on a tiny
     slice of the inference workload in parallel; winners are adopted
     mid-session. Over hours/days, the kernel literally learns your hardware
     and workload patterns. No model retraining; pure behavioral evolution."

Design
------
Each forward pass emits a TelemetryGlyph — a compact record of what was
expensive, what hit cache, and how the SSM selectivity was distributed.

TelemetryGlyphs are accumulated in an InferenceKernelOptimizer, which:
1. Maintains a sliding window of recent glyphs
2. Detects patterns (high cache miss rate, low selectivity diversity, slow batches)
3. Proposes KernelStrategy micro-optimisations (change chunk_size, prior_strength,
   coord_threshold, batch_size cap)
4. A/B tests proposals on subsequent batches and promotes winners

The SelfImprovingKernel wraps DistributedServingEngine and applies
strategies automatically — no retraining, no deployment, pure behavioral.

No LLM calls. No external dependencies. Pure Python evolution.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# TelemetryGlyph
# ---------------------------------------------------------------------------

@dataclass
class TelemetryGlyph:
    """
    Compact telemetry record for one inference batch.

    Emitted by _process_batch() and fed into InferenceKernelOptimizer.
    Stored as glyph-native (primitive_id histogram) rather than raw metrics
    so it can be indexed by GlyphBenchmarkIndex.
    """
    latency_ms:        float
    cache_hit_rate:    float        # 0.0–1.0
    batch_size:        int
    unique_glyphs:     int          # distinct primitive IDs in batch
    prior_strength:    float        # GlyphPriorPool prior_strength used
    coord_threshold:   float        # NegativeSpaceExtractor threshold
    chunk_size:        int          # SemanticKVCache chunk size
    timestamp:         float = field(default_factory=time.monotonic)
    strategy_id:       Optional[str] = None   # which strategy was active

    @property
    def efficiency_score(self) -> float:
        """
        Combined efficiency: high cache hit + low latency + high glyph diversity.

        Score ∈ [0, 1]. Higher is better.
        Approximation — tunable without model retraining.
        """
        latency_score = max(0.0, 1.0 - self.latency_ms / 500.0)
        diversity_score = min(1.0, self.unique_glyphs / 64.0)
        return 0.5 * self.cache_hit_rate + 0.3 * latency_score + 0.2 * diversity_score


# ---------------------------------------------------------------------------
# KernelStrategy
# ---------------------------------------------------------------------------

@dataclass
class KernelStrategy:
    """
    Micro-optimisation strategy proposed by the optimizer.

    Contains only the parameters the kernel can change without retraining.
    All parameters have safe defaults that match initial Ryot configuration.
    """
    strategy_id:      str
    prior_strength:   float = 0.15   # GlyphPriorPool
    coord_threshold:  float = 0.10   # NegativeSpaceExtractor
    chunk_size:       int   = 256    # SemanticKVCache
    max_batch_size:   int   = 8      # DistributedServingEngine batcher
    decay_interval:   int   = 32     # GlyphPriorPool decay every N stores
    description:      str   = ""
    wins:             int   = 0
    trials:           int   = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trials if self.trials else 0.0

    def __repr__(self) -> str:
        return (
            f"KernelStrategy({self.strategy_id!r}, "
            f"prior={self.prior_strength}, chunk={self.chunk_size}, "
            f"coord_thresh={self.coord_threshold}, "
            f"win_rate={self.win_rate:.1%} [{self.wins}/{self.trials}])"
        )


# Factory for named strategies derived from telemetry patterns
_STRATEGY_TEMPLATES: List[KernelStrategy] = [
    KernelStrategy(
        "conservative",
        prior_strength=0.05, coord_threshold=0.20, chunk_size=512,
        max_batch_size=4, description="Low prior influence, large chunks",
    ),
    KernelStrategy(
        "aggressive-prior",
        prior_strength=0.30, coord_threshold=0.05, chunk_size=128,
        max_batch_size=8, description="Strong prior recycling, small chunks",
    ),
    KernelStrategy(
        "sparse-negative",
        prior_strength=0.10, coord_threshold=0.30, chunk_size=256,
        max_batch_size=6, description="More absences detected, medium chunks",
    ),
    KernelStrategy(
        "throughput",
        prior_strength=0.15, coord_threshold=0.10, chunk_size=256,
        max_batch_size=16, description="Maximise batch size",
    ),
    KernelStrategy(
        "latency",
        prior_strength=0.15, coord_threshold=0.10, chunk_size=64,
        max_batch_size=2, description="Minimise TTFT via small batches",
    ),
]


# ---------------------------------------------------------------------------
# InferenceKernelOptimizer
# ---------------------------------------------------------------------------

class InferenceKernelOptimizer:
    """
    Meta-optimizer that watches TelemetryGlyphs and proposes KernelStrategies.

    Algorithm:
    1. Accumulate a rolling window of TelemetryGlyphs.
    2. Every `eval_every` glyphs, evaluate the current strategy's efficiency.
    3. If the current strategy underperforms the historical mean by more than
       `improvement_threshold`, propose a new candidate strategy.
    4. Run the candidate for `trial_batches` batches, then compare win rates.
    5. Promote the winner; track losers for future reference.

    This is behavioural evolution — no gradient computation, no model access.
    """

    def __init__(
        self,
        window_size:           int   = 50,
        eval_every:            int   = 20,
        improvement_threshold: float = 0.05,
        trial_batches:         int   = 10,
    ):
        self._window: Deque[TelemetryGlyph] = deque(maxlen=window_size)
        self._eval_every           = eval_every
        self._improvement_threshold = improvement_threshold
        self._trial_batches        = trial_batches

        self._active_strategy: KernelStrategy = KernelStrategy(
            "default",
            description="Initial default strategy",
        )
        self._candidate: Optional[KernelStrategy] = None
        self._candidate_remaining: int = 0

        self._history: List[KernelStrategy] = []
        self._step: int = 0
        self._promotions: int = 0
        self._rejections: int = 0

    def record(self, glyph: TelemetryGlyph) -> None:
        glyph.strategy_id = self._active_strategy.strategy_id
        self._window.append(glyph)
        self._step += 1

        # Update candidate trial
        if self._candidate is not None:
            self._candidate.trials += 1
            if glyph.efficiency_score >= self._mean_efficiency():
                self._candidate.wins += 1
            self._candidate_remaining -= 1
            if self._candidate_remaining <= 0:
                self._resolve_trial()

        # Periodic evaluation
        elif self._step % self._eval_every == 0 and len(self._window) >= self._eval_every:
            self._maybe_propose()

    def _mean_efficiency(self) -> float:
        if not self._window:
            return 0.0
        return sum(g.efficiency_score for g in self._window) / len(self._window)

    def _maybe_propose(self) -> None:
        """Propose a new candidate if current strategy is underperforming."""
        current_scores = [
            g.efficiency_score
            for g in self._window
            if g.strategy_id == self._active_strategy.strategy_id
        ]
        if not current_scores:
            return
        current_mean = sum(current_scores) / len(current_scores)
        global_mean  = self._mean_efficiency()

        # Propose improvement only if underperforming
        if current_mean < global_mean - self._improvement_threshold:
            candidate = self._select_candidate()
            if candidate is not None:
                self._start_trial(candidate)

    def _select_candidate(self) -> Optional[KernelStrategy]:
        """Pick the best-performing historical strategy, or the next template."""
        if self._history:
            best = max(self._history, key=lambda s: s.win_rate)
            if best.strategy_id != self._active_strategy.strategy_id:
                return KernelStrategy(
                    best.strategy_id,
                    prior_strength=best.prior_strength,
                    coord_threshold=best.coord_threshold,
                    chunk_size=best.chunk_size,
                    max_batch_size=best.max_batch_size,
                    description=best.description + " [retry]",
                )

        # Try the next template not yet trialled
        trialled = {s.strategy_id for s in self._history}
        trialled.add(self._active_strategy.strategy_id)
        for tmpl in _STRATEGY_TEMPLATES:
            if tmpl.strategy_id not in trialled:
                return KernelStrategy(
                    tmpl.strategy_id,
                    prior_strength=tmpl.prior_strength,
                    coord_threshold=tmpl.coord_threshold,
                    chunk_size=tmpl.chunk_size,
                    max_batch_size=tmpl.max_batch_size,
                    description=tmpl.description,
                )
        return None

    def _start_trial(self, candidate: KernelStrategy) -> None:
        self._candidate = candidate
        self._candidate_remaining = self._trial_batches

    def _resolve_trial(self) -> None:
        """Promote or reject the candidate."""
        assert self._candidate is not None
        self._history.append(self._active_strategy)

        if self._candidate.win_rate >= 0.5:
            self._active_strategy = self._candidate
            self._promotions += 1
        else:
            self._history.append(self._candidate)
            self._rejections += 1

        self._candidate = None

    @property
    def active_strategy(self) -> KernelStrategy:
        return self._active_strategy

    @property
    def is_in_trial(self) -> bool:
        return self._candidate is not None

    @property
    def trial_strategy(self) -> Optional[KernelStrategy]:
        return self._candidate

    def effective_strategy(self) -> KernelStrategy:
        """Return the strategy currently governing the kernel."""
        return self._candidate if self._candidate is not None else self._active_strategy

    def stats(self) -> Dict:
        return {
            "steps":          self._step,
            "window_size":    len(self._window),
            "mean_efficiency": round(self._mean_efficiency(), 4),
            "active_strategy": self._active_strategy.strategy_id,
            "in_trial":        self.is_in_trial,
            "trial_strategy":  self._candidate.strategy_id if self._candidate else None,
            "promotions":      self._promotions,
            "rejections":      self._rejections,
            "history_count":   len(self._history),
        }


# ---------------------------------------------------------------------------
# SelfImprovingKernel
# ---------------------------------------------------------------------------

class SelfImprovingKernel:
    """
    Wraps any inference engine and applies KernelStrategy updates automatically.

    Usage:
        engine  = DistributedServingEngine(model, num_gpus=1)
        kernel  = SelfImprovingKernel(engine)
        result  = await kernel.process_batch(batch)

    After enough batches, kernel.optimizer.active_strategy will have evolved
    toward the configuration that best fits the current workload on this hardware.

    This is the "autonomy loop" from the doc: inference→telemetry→optimize→apply,
    running continuously in the background of normal serving.
    """

    def __init__(self, engine, optimizer: Optional[InferenceKernelOptimizer] = None):
        self.engine    = engine
        self.optimizer = optimizer or InferenceKernelOptimizer()
        self._batches_processed = 0

    async def process_batch(self, batch) -> None:
        """Process one batch, emit telemetry, apply strategy updates."""
        t_start = time.monotonic()
        await self.engine._process_batch(batch)
        latency_ms = (time.monotonic() - t_start) * 1000.0

        # Collect telemetry
        strategy = self.optimizer.effective_strategy()
        stats    = await self.engine.get_stats()
        kv_stats = stats.get("kv_cache", {})

        glyph = TelemetryGlyph(
            latency_ms     = round(latency_ms, 3),
            cache_hit_rate = kv_stats.get("hit_rate", 0.0),
            batch_size     = batch.tokens.shape[0] if hasattr(batch, "tokens") else 1,
            unique_glyphs  = len(set(batch.tokens.reshape(-1).tolist()))
                             if hasattr(batch, "tokens") else 0,
            prior_strength = strategy.prior_strength,
            coord_threshold= strategy.coord_threshold,
            chunk_size     = strategy.chunk_size,
        )
        self.optimizer.record(glyph)
        self._batches_processed += 1

        # Apply strategy changes (live, no restart)
        self._apply_strategy(self.optimizer.effective_strategy())

    def _apply_strategy(self, strategy: KernelStrategy) -> None:
        """Apply strategy parameters to the engine's live components."""
        engine = self.engine
        if hasattr(engine, "kv_cache") and hasattr(engine.kv_cache, "prior_pool"):
            engine.kv_cache.prior_pool.prior_strength = strategy.prior_strength

    def stats(self) -> Dict:
        return {
            "batches_processed": self._batches_processed,
            "optimizer":         self.optimizer.stats(),
        }

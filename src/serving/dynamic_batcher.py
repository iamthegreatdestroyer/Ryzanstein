"""
src/serving/dynamic_batcher.py — Dynamic Request Batching for Ryzanstein LLM

Sprint 7.3: Dynamic Batching
Autonomy Level: 75%

Accumulates concurrent inference requests into optimal batches:
- Token-budget-aware (avoids OOM by capping total tokens per batch)
- Priority-aware (SLA tier 0 = interactive, tier 1 = batch)
- Deadline-aware (flush when the earliest deadline is exceeded)
- Configurable max_wait_ms for latency vs. throughput trade-off
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchRequest:
    """A single inference request waiting to be batched.

    Attributes:
        request_id: Unique request identifier for tracing.
        prompt: The text prompt to generate from.
        max_tokens: Maximum tokens to generate.
        priority: Lower is higher priority. 0 = interactive, 1 = batch.
        deadline: Absolute wall-clock deadline (time.monotonic() seconds).
        future: Asyncio Future resolved with the inference result.
    """

    prompt: str
    max_tokens: int = 256
    priority: int = 0
    deadline: float = field(default_factory=lambda: time.monotonic() + 30.0)
    request_id: str = field(default_factory=lambda: uuid4().hex[:8])
    future: asyncio.Future = field(default_factory=asyncio.get_event_loop().create_future)

    def estimated_tokens(self) -> int:
        """Rough token estimate: words * 1.3 + max_tokens."""
        words = len(self.prompt.split())
        return int(words * 1.3) + self.max_tokens


@dataclass
class BatchResult:
    """Result for a single request within a batch."""

    request_id: str
    text: str
    tokens_generated: int
    generation_time_ms: float
    batch_size: int  # how many requests were in the same batch


# ─────────────────────────────────────────────────────────────────────────────
# Batcher Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatcherConfig:
    max_batch_size: int = 32
    max_wait_ms: float = 20.0           # max latency added by waiting
    max_tokens_per_batch: int = 4096    # token budget per batch (OOM guard)
    priority_levels: int = 2            # 0 = interactive, 1 = background
    enable_stats: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Batcher
# ─────────────────────────────────────────────────────────────────────────────

class DynamicBatcher:
    """
    Accumulates requests and dispatches optimal batches to the inference engine.

    Strategy:
    1. Requests enter via ``submit()``; each gets a Future.
    2. Background loop ``run()`` flushes when:
       - ``max_batch_size`` is reached (immediate flush)
       - Token budget ``max_tokens_per_batch`` is reached (immediate flush)
       - ``max_wait_ms`` elapses without a full batch (timeout flush)
       - The earliest request deadline is exceeded (deadline flush)
    3. Requests are sorted by (-priority, deadline) before dispatch.
    4. The inference function receives a list of prompts; results are zipped
       back to waiting Futures.

    Usage::

        batcher = DynamicBatcher(config=BatcherConfig(max_wait_ms=10))

        async def inference(prompts: list[str]) -> list[str]:
            return await engine.batch_generate(prompts)

        # Start the background flush loop
        asyncio.create_task(batcher.run(inference))

        # Submit requests from request handlers
        result: BatchResult = await batcher.submit(
            BatchRequest(prompt="Hello world", priority=0)
        )
    """

    def __init__(self, config: Optional[BatcherConfig] = None) -> None:
        self.config = config or BatcherConfig()
        self._queue: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._stats = {
            "batches_dispatched": 0,
            "requests_served": 0,
            "total_wait_ms": 0.0,
            "immediate_flushes": 0,
            "timeout_flushes": 0,
            "deadline_flushes": 0,
        }
        self._running = False

    # ── Public API ──

    async def submit(self, request: BatchRequest) -> BatchResult:
        """
        Submit a request and await its result.

        The returned awaitable resolves when the request has been processed as
        part of a batch. The Future is created lazily to avoid event-loop
        issues when DynamicBatcher is instantiated at module level.
        """
        loop = asyncio.get_event_loop()
        request.future = loop.create_future()

        async with self._lock:
            self._queue.append(request)
            should_flush = self._should_flush_immediately()

        if should_flush:
            self._flush_event.set()

        return await request.future

    async def run(
        self,
        inference_fn: Callable[[List[str]], Coroutine[Any, Any, List[str]]],
    ) -> None:
        """
        Background flush loop. Run this as a long-lived asyncio task.

        ``inference_fn`` receives a list of prompt strings and must return
        a list of result strings in the same order.
        """
        self._running = True
        wait_s = self.config.max_wait_ms / 1000.0

        while self._running:
            try:
                await asyncio.wait_for(self._flush_event.wait(), timeout=wait_s)
                flush_reason = "size"
            except asyncio.TimeoutError:
                flush_reason = "timeout"
                self._stats["timeout_flushes"] += 1

            self._flush_event.clear()

            async with self._lock:
                if not self._queue:
                    continue

                # Check for deadline violations
                now = time.monotonic()
                if any(r.deadline <= now for r in self._queue):
                    flush_reason = "deadline"
                    self._stats["deadline_flushes"] += 1

                if flush_reason == "size":
                    self._stats["immediate_flushes"] += 1

                # Sort: highest priority (0) first, then earliest deadline
                batch = sorted(self._queue, key=lambda r: (r.priority, r.deadline))
                self._queue.clear()

            await self._dispatch_batch(batch, inference_fn)

    async def stop(self) -> None:
        """Signal the run loop to stop after the current batch."""
        self._running = False
        self._flush_event.set()

    # ── Internal ──

    def _should_flush_immediately(self) -> bool:
        """Return True if the current queue warrants an immediate flush."""
        if len(self._queue) >= self.config.max_batch_size:
            return True
        total_tokens = sum(r.estimated_tokens() for r in self._queue)
        if total_tokens >= self.config.max_tokens_per_batch:
            return True
        now = time.monotonic()
        if any(r.deadline <= now for r in self._queue):
            return True
        return False

    async def _dispatch_batch(
        self,
        batch: List[BatchRequest],
        inference_fn: Callable[[List[str]], Coroutine[Any, Any, List[str]]],
    ) -> None:
        """Run ``inference_fn`` and resolve all Futures in the batch."""
        if not batch:
            return

        batch_size = len(batch)
        prompts = [r.prompt for r in batch]
        start = time.monotonic()

        try:
            raw_results = await inference_fn(prompts)
            duration_ms = (time.monotonic() - start) * 1000.0

            for req, text in zip(batch, raw_results):
                est_tokens = req.estimated_tokens()
                result = BatchResult(
                    request_id=req.request_id,
                    text=text,
                    tokens_generated=est_tokens,
                    generation_time_ms=duration_ms,
                    batch_size=batch_size,
                )
                if not req.future.done():
                    req.future.set_result(result)

            self._stats["batches_dispatched"] += 1
            self._stats["requests_served"] += batch_size
            self._stats["total_wait_ms"] += duration_ms
            logger.debug(
                "Batch dispatched: size=%d tokens≈%d duration=%.1fms",
                batch_size,
                sum(r.estimated_tokens() for r in batch),
                duration_ms,
            )

        except Exception as exc:
            logger.error("Batch inference failed: %s", exc, exc_info=True)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

    def stats(self) -> dict:
        avg_wait = (
            self._stats["total_wait_ms"] / max(self._stats["batches_dispatched"], 1)
        )
        return {**self._stats, "avg_batch_wait_ms": round(avg_wait, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Priority queue variant for SLA tiers
# ─────────────────────────────────────────────────────────────────────────────

class TieredBatcher:
    """
    Two-queue batcher that keeps interactive (tier 0) and batch (tier 1)
    requests in separate queues so SLA-critical requests never wait behind
    long-running batch jobs.

    The interactive batcher runs with a tight max_wait_ms (e.g. 5ms).
    The background batcher uses a relaxed window (e.g. 100ms).
    Both share the same ``inference_fn``.
    """

    def __init__(
        self,
        interactive_config: Optional[BatcherConfig] = None,
        background_config: Optional[BatcherConfig] = None,
    ) -> None:
        self._interactive = DynamicBatcher(
            interactive_config or BatcherConfig(max_wait_ms=5.0, max_batch_size=8)
        )
        self._background = DynamicBatcher(
            background_config or BatcherConfig(max_wait_ms=100.0, max_batch_size=32)
        )

    async def submit(self, request: BatchRequest) -> BatchResult:
        if request.priority == 0:
            return await self._interactive.submit(request)
        return await self._background.submit(request)

    async def run(
        self,
        inference_fn: Callable[[List[str]], Coroutine[Any, Any, List[str]]],
    ) -> None:
        await asyncio.gather(
            self._interactive.run(inference_fn),
            self._background.run(inference_fn),
        )

    def stats(self) -> dict:
        return {
            "interactive": self._interactive.stats(),
            "background": self._background.stats(),
        }

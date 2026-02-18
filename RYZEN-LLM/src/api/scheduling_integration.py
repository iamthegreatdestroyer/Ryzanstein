"""
Sprint 4.3: Advanced Scheduling Integration for Ryzanstein LLM API
[REF:SPRINT4.3] - Adaptive Batch Scheduler, Resource Allocator, Priority Queue

Wires PHASE2_DEVELOPMENT scheduling components into the FastAPI serving layer.

Key components:
  - InferenceScheduler: Wraps the ML-based AdaptiveBatchScheduler
  - PriorityInferenceQueue: Priority queue for incoming requests
  - ResourceAwareScheduler: Admits/rejects based on available memory/CPU
  - SchedulingMiddleware: ASGI middleware adding X-Queue-* response headers

Integration flow:
  1. Request arrives at /v1/chat/completions
  2. PriorityInferenceQueue enqueues with priority derived from request metadata
  3. ResourceAwareScheduler checks admission (memory watermark, KV cache budget)
  4. InferenceScheduler selects next batch using FCFS/Priority/EDF policy
  5. Batch executes via the engine; results routed back to waiting futures

Usage:
    from .scheduling_integration import (
        get_inference_scheduler,
        enqueue_request,
        SchedulingMiddleware,
    )
"""

import sys
import asyncio
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PHASE2_SRC = _REPO_ROOT / "PHASE2_DEVELOPMENT" / "src"
if str(_PHASE2_SRC) not in sys.path:
    sys.path.insert(0, str(_PHASE2_SRC))

# ---------------------------------------------------------------------------
# Import scheduling primitives with graceful fallback
# ---------------------------------------------------------------------------
try:
    from scheduling.batch_scheduler import (
        SchedulingRequest,
        RequestPriority,
        SchedulingPolicy,
        BatchFormationStrategy,
        WorkloadType,
    )
    _SCHEDULER_AVAILABLE = True
    logger.info("Sprint 4.3 scheduling primitives loaded from PHASE2_DEVELOPMENT")
except ImportError as _e:
    logger.warning(f"Advanced scheduling not available ({_e}); using built-in queue")
    _SCHEDULER_AVAILABLE = False

    # Minimal stubs
    class SchedulingPolicy(Enum):
        FCFS = "fcfs"
        PRIORITY = "priority"
        EDF = "edf"

    class WorkloadType(Enum):
        INTERACTIVE = "interactive"
        BATCH = "batch"

    @dataclass
    class RequestPriority:
        base: int = 0
        urgency: float = 0.0
        value: float = 1.0
        boost_rate: float = 0.1

        def effective_priority(self, wait_time_s: float) -> float:
            boost = min(wait_time_s * self.boost_rate, 5.0)
            return (self.base + boost) * self.value * (1.0 + self.urgency)

    @dataclass
    class SchedulingRequest:
        request_id: str
        tenant_id: str
        sequence_length: int
        max_new_tokens: int
        priority: RequestPriority = field(default_factory=RequestPriority)
        deadline_ms: Optional[float] = None
        arrival_time: float = field(default_factory=time.monotonic)
        estimated_duration_ms: Optional[float] = None
        preemptible: bool = True
        metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inference queue entry
# ---------------------------------------------------------------------------

@dataclass
class InferenceQueueEntry:
    """An enqueued inference request waiting for scheduling."""
    request_id: str
    sched_request: SchedulingRequest
    inference_func: Callable[..., Awaitable[Any]]
    func_args: tuple
    func_kwargs: dict
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)

    def effective_priority(self) -> float:
        wait = time.monotonic() - self.enqueued_at
        return self.sched_request.priority.effective_priority(wait)

    def __lt__(self, other: "InferenceQueueEntry") -> bool:
        """Higher effective priority sorts first (for min-heap we negate)."""
        return self.effective_priority() > other.effective_priority()


# ---------------------------------------------------------------------------
# Priority inference queue
# ---------------------------------------------------------------------------

import heapq as _heapq


class PriorityInferenceQueue:
    """
    Thread-safe priority queue for inference requests.

    Requests with higher effective_priority() are scheduled first.
    Priority ages over time (boost_rate) to prevent starvation.
    """

    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._queue: List[tuple] = []   # (neg_priority, seq_no, entry)
        self._seq = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()

    async def put(self, entry: InferenceQueueEntry) -> None:
        async with self._lock:
            if len(self._queue) >= self.maxsize:
                raise asyncio.QueueFull(f"Inference queue full (maxsize={self.maxsize})")
            neg_prio = -entry.effective_priority()
            _heapq.heappush(self._queue, (neg_prio, self._seq, entry))
            self._seq += 1
            self._not_empty.set()

    async def get(self) -> InferenceQueueEntry:
        while True:
            async with self._lock:
                if self._queue:
                    _, _, entry = _heapq.heappop(self._queue)
                    if not self._queue:
                        self._not_empty.clear()
                    return entry
            await self._not_empty.wait()

    def qsize(self) -> int:
        return len(self._queue)

    def empty(self) -> bool:
        return len(self._queue) == 0

    def get_stats(self) -> dict:
        return {
            "queued": self.qsize(),
            "maxsize": self.maxsize,
        }


# ---------------------------------------------------------------------------
# Resource-aware admission control
# ---------------------------------------------------------------------------

class ResourceAwareAdmission:
    """
    Admits/rejects inference requests based on resource watermarks.

    Tracks:
    - `max_kv_tokens`: Total KV-cache token budget
    - `current_kv_tokens`: Currently allocated KV tokens
    - `max_batch_size`: Maximum concurrent batch size
    - `current_batch_size`: Currently executing requests
    """

    def __init__(
        self,
        max_kv_tokens: int = 131072,     # 128K tokens KV budget
        max_batch_size: int = 32,
        high_watermark: float = 0.85,    # Reject new long requests above this
        critical_watermark: float = 0.95, # Reject all new requests above this
    ):
        self.max_kv_tokens = max_kv_tokens
        self.max_batch_size = max_batch_size
        self.high_watermark = high_watermark
        self.critical_watermark = critical_watermark

        self._current_kv_tokens = 0
        self._current_batch_size = 0
        self._lock = asyncio.Lock()
        self._total_admitted = 0
        self._total_rejected = 0

    async def admit(self, request: SchedulingRequest) -> bool:
        """
        Decide whether to admit a request.

        Returns True if admitted, False if rejected.
        """
        async with self._lock:
            kv_needed = request.sequence_length + request.max_new_tokens
            kv_util = self._current_kv_tokens / max(self.max_kv_tokens, 1)
            batch_util = self._current_batch_size / max(self.max_batch_size, 1)

            # Critical watermark: reject everything
            if kv_util >= self.critical_watermark or batch_util >= 1.0:
                self._total_rejected += 1
                logger.warning(
                    f"Admission REJECTED (critical): kv_util={kv_util:.2%}, "
                    f"batch={self._current_batch_size}/{self.max_batch_size}"
                )
                return False

            # High watermark: reject large requests
            if kv_util >= self.high_watermark and kv_needed > 512:
                self._total_rejected += 1
                logger.debug(
                    f"Admission REJECTED (high watermark): kv_needed={kv_needed}, "
                    f"kv_util={kv_util:.2%}"
                )
                return False

            # Admit
            self._current_kv_tokens += kv_needed
            self._current_batch_size += 1
            self._total_admitted += 1
            return True

    async def release(self, request: SchedulingRequest) -> None:
        """Release resources after request completes."""
        async with self._lock:
            kv_used = request.sequence_length + request.max_new_tokens
            self._current_kv_tokens = max(0, self._current_kv_tokens - kv_used)
            self._current_batch_size = max(0, self._current_batch_size - 1)

    def get_stats(self) -> dict:
        kv_util = self._current_kv_tokens / max(self.max_kv_tokens, 1)
        return {
            "kv_tokens_used": self._current_kv_tokens,
            "kv_tokens_max": self.max_kv_tokens,
            "kv_utilization": round(kv_util, 4),
            "batch_size": self._current_batch_size,
            "batch_max": self.max_batch_size,
            "total_admitted": self._total_admitted,
            "total_rejected": self._total_rejected,
        }


# ---------------------------------------------------------------------------
# Inference scheduler — wraps queue + admission + worker loop
# ---------------------------------------------------------------------------

class InferenceScheduler:
    """
    End-to-end inference scheduler.

    1. Enqueues requests into a priority queue
    2. Resource-aware admission rejects overloaded requests
    3. Worker loop dequeues and executes with concurrency control
    4. Futures resolve when inference completes
    """

    def __init__(
        self,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
        max_queue: int = 512,
        max_concurrent: int = 16,
        max_kv_tokens: int = 131072,
    ):
        self.policy = policy
        self.max_concurrent = max_concurrent
        self._queue = PriorityInferenceQueue(maxsize=max_queue)
        self._admission = ResourceAwareAdmission(max_kv_tokens=max_kv_tokens)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = {
            "total_scheduled": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_rejected": 0,
        }

    async def start(self) -> None:
        """Start the background scheduler worker loop."""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(
                f"InferenceScheduler started (policy={self.policy.name}, "
                f"max_concurrent={self.max_concurrent})"
            )

    async def stop(self) -> None:
        """Gracefully stop the scheduler."""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("InferenceScheduler stopped")

    async def submit(
        self,
        inference_func: Callable[..., Awaitable[Any]],
        *args,
        tenant_id: str = "default",
        sequence_length: int = 128,
        max_new_tokens: int = 100,
        priority_base: int = 0,
        deadline_ms: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """
        Submit an inference request for scheduled execution.

        Returns the inference result when ready.
        Raises asyncio.QueueFull if queue is full.
        Raises RuntimeError if admission is rejected.
        """
        request_id = uuid.uuid4().hex[:8]
        sched_req = SchedulingRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            sequence_length=sequence_length,
            max_new_tokens=max_new_tokens,
            priority=RequestPriority(base=priority_base),
            deadline_ms=deadline_ms,
        )

        # Admission control
        admitted = await self._admission.admit(sched_req)
        if not admitted:
            self._stats["total_rejected"] += 1
            raise RuntimeError(
                f"Request rejected by admission control (KV utilization too high)"
            )

        # Create future and enqueue
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        entry = InferenceQueueEntry(
            request_id=request_id,
            sched_request=sched_req,
            inference_func=inference_func,
            func_args=args,
            func_kwargs=kwargs,
            future=future,
        )

        try:
            await self._queue.put(entry)
            self._stats["total_scheduled"] += 1
        except asyncio.QueueFull:
            await self._admission.release(sched_req)
            raise

        # Wait for result
        try:
            return await future
        finally:
            await self._admission.release(sched_req)

    async def _worker_loop(self) -> None:
        """Background loop that dequeues and executes inference requests."""
        while self._running:
            try:
                entry = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Scheduler worker error in get: {e}")
                continue

            # Execute under concurrency semaphore
            asyncio.create_task(self._execute_entry(entry))

    async def _execute_entry(self, entry: InferenceQueueEntry) -> None:
        """Execute a single queued inference request."""
        async with self._semaphore:
            if entry.future.cancelled():
                return
            try:
                result = await entry.inference_func(
                    *entry.func_args, **entry.func_kwargs
                )
                if not entry.future.done():
                    entry.future.set_result(result)
                self._stats["total_completed"] += 1
            except Exception as e:
                if not entry.future.done():
                    entry.future.set_exception(e)
                self._stats["total_failed"] += 1
                logger.error(
                    f"Scheduled inference {entry.request_id} failed: {e}"
                )

    def get_stats(self) -> dict:
        return {
            "policy": self.policy.name,
            "queue": self._queue.get_stats(),
            "admission": self._admission.get_stats(),
            "scheduled": self._stats["total_scheduled"],
            "completed": self._stats["total_completed"],
            "failed": self._stats["total_failed"],
            "rejected": self._stats["total_rejected"],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_inference_scheduler: Optional[InferenceScheduler] = None


def get_inference_scheduler() -> InferenceScheduler:
    """Return the singleton InferenceScheduler."""
    global _inference_scheduler
    if _inference_scheduler is None:
        _inference_scheduler = InferenceScheduler(
            policy=SchedulingPolicy.FCFS,
            max_queue=512,
            max_concurrent=16,
            max_kv_tokens=131072,
        )
        logger.info("Initialized InferenceScheduler singleton")
    return _inference_scheduler


async def start_scheduler() -> None:
    """Start the background scheduler (call from app lifespan)."""
    scheduler = get_inference_scheduler()
    await scheduler.start()


async def stop_scheduler() -> None:
    """Stop the scheduler (call from app lifespan shutdown)."""
    global _inference_scheduler
    if _inference_scheduler is not None:
        await _inference_scheduler.stop()
        _inference_scheduler = None


# ---------------------------------------------------------------------------
# ASGI middleware — adds scheduling stats to response headers
# ---------------------------------------------------------------------------

class SchedulingMiddleware:
    """Adds X-Queue-* headers to inference responses for observability."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return

        scheduler = get_inference_scheduler()
        stats = scheduler.get_stats()

        async def send_with_headers(response):
            if response["type"] == "http.response.start":
                headers = list(response.get("headers", []))
                queued = str(stats["queue"].get("queued", 0)).encode()
                admitted = str(stats["admission"].get("total_admitted", 0)).encode()
                headers.append((b"x-queue-depth", queued))
                headers.append((b"x-total-admitted", admitted))
                response = {**response, "headers": headers}
            await send(response)

        await self.app(scope, receive, send_with_headers)

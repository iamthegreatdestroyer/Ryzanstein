"""
Request Batching Engine for Distributed Inference
=================================================

Implements dynamic batching to optimize GPU utilization and throughput.
Groups requests by sequence length and schedules them efficiently.

Key Features:
- Dynamic batch formation based on sequence similarity
- Memory-efficient padding strategies
- Batch timeout handling for latency guarantees
- Priority-based scheduling (FCFS with optional priority)
- Batch size optimization for GPU efficiency

Architecture:
- Request queue with priority ordering
- Batch formation algorithm
- Memory-aware padding
- Timeout management for latency bounds
"""

import asyncio
import time
import logging
from typing import List, Optional, Dict, Any, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Request batching priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchedRequest:
    """A request ready for batch processing."""
    request_id: str
    gpu_id: int
    priority: BatchPriority
    input_tokens: List[int]
    max_new_tokens: int
    temperature: float
    timestamp: float
    timeout_at: Optional[float] = None

    # Response handling
    future: Optional[asyncio.Future] = None


@dataclass
class Batch:
    """A batch of requests for processing."""
    batch_id: str
    gpu_id: int
    requests: List[BatchedRequest] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    max_sequence_length: int = 0
    total_tokens: int = 0

    def add_request(self, request: BatchedRequest):
        """Add a request to this batch."""
        self.requests.append(request)
        self.max_sequence_length = max(self.max_sequence_length, len(request.input_tokens))
        self.total_tokens += len(request.input_tokens)

    def get_batch_size(self) -> int:
        """Get the number of requests in this batch."""
        return len(self.requests)

    def get_memory_estimate(self) -> int:
        """Estimate memory usage for this batch (in tokens)."""
        # Account for padding to max sequence length
        return self.max_sequence_length * self.get_batch_size()

    def should_process(self, max_batch_size: int, max_latency_ms: float) -> bool:
        """Check if batch should be processed."""
        current_time = time.time() * 1000  # Convert to ms

        # Check batch size limit
        if self.get_batch_size() >= max_batch_size:
            return True

        # Check latency limit
        for request in self.requests:
            if request.timeout_at and current_time >= request.timeout_at:
                return True

        return False


class BatchEngine:
    """
    Engine for dynamic request batching and scheduling.

    Optimizes GPU utilization by grouping similar requests together
    while maintaining latency guarantees.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_latency_ms: float = 100.0,
        batch_timeout_ms: float = 50.0,
        similarity_threshold: float = 0.8
    ):
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.batch_timeout_ms = batch_timeout_ms
        self.similarity_threshold = similarity_threshold

        # Request queues per GPU
        self.request_queues: Dict[int, Deque[BatchedRequest]] = {}
        self.active_batches: Dict[int, Batch] = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
            "timeout_events": 0,
            "efficiency_ratio": 0.0
        }

        # Threading
        self.lock = threading.Lock()
        self._shutdown_event = threading.Event()

    def initialize_gpu_queues(self, gpu_ids: List[int]):
        """Initialize request queues for GPUs."""
        for gpu_id in gpu_ids:
            self.request_queues[gpu_id] = deque()
            self.active_batches[gpu_id] = Batch(
                batch_id=f"batch_{gpu_id}_{int(time.time())}",
                gpu_id=gpu_id
            )

    async def submit_request(
        self,
        request_id: str,
        gpu_id: int,
        input_tokens: List[int],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        priority: BatchPriority = BatchPriority.NORMAL
    ) -> asyncio.Future:
        """
        Submit a request for batching.

        Returns a Future that will be completed when the request is processed.
        """
        future = asyncio.Future()

        batched_request = BatchedRequest(
            request_id=request_id,
            gpu_id=gpu_id,
            priority=priority,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timestamp=time.time(),
            timeout_at=time.time() * 1000 + self.max_latency_ms,  # Convert to ms
            future=future
        )

        with self.lock:
            if gpu_id not in self.request_queues:
                self.initialize_gpu_queues([gpu_id])

            # Add to appropriate queue based on priority
            self._enqueue_request(batched_request)
            self.stats["total_requests"] += 1

        # Try to form batches immediately
        await self._try_form_batches()

        return future

    def _enqueue_request(self, request: BatchedRequest):
        """Enqueue request in priority order."""
        queue = self.request_queues[request.gpu_id]

        # Insert based on priority (higher priority first)
        insert_pos = 0
        for i, existing in enumerate(queue):
            if request.priority.value > existing.priority.value:
                insert_pos = i
                break
            insert_pos = i + 1

        queue.insert(insert_pos, request)

    async def _try_form_batches(self):
        """Try to form new batches from queued requests."""
        for gpu_id, queue in self.request_queues.items():
            if not queue:
                continue

            batch = self.active_batches[gpu_id]

            # Add requests to batch until full or timeout
            while queue and not batch.should_process(self.max_batch_size, self.max_latency_ms):
                request = queue.popleft()
                batch.add_request(request)

            # Process batch if ready
            if batch.should_process(self.max_batch_size, self.max_latency_ms):
                await self._process_batch(batch)

                # Create new batch for this GPU
                self.active_batches[gpu_id] = Batch(
                    batch_id=f"batch_{gpu_id}_{int(time.time())}",
                    gpu_id=gpu_id
                )

    async def _process_batch(self, batch: Batch):
        """Process a completed batch."""
        start_time = time.time()

        try:
            logger.debug(f"Processing batch {batch.batch_id} with {batch.get_batch_size()} requests")

            # In a real implementation, this would forward to GPU worker
            # For now, simulate processing
            await self._simulate_batch_processing(batch)

            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_batch_stats(batch, processing_time)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fail all requests in batch
            for request in batch.requests:
                if request.future and not request.future.done():
                    request.future.set_exception(e)

    async def _simulate_batch_processing(self, batch: Batch):
        """Simulate batch processing (replace with actual GPU processing)."""
        # Simulate processing time based on batch size
        processing_time = 0.01 * batch.get_batch_size()  # 10ms per request
        await asyncio.sleep(processing_time)

        # Generate mock responses
        for request in batch.requests:
            if request.future and not request.future.done():
                # Mock response
                response = {
                    "request_id": request.request_id,
                    "generated_tokens": [1, 2, 3] * (request.max_new_tokens // 3),  # Mock tokens
                    "finish_reason": "length" if len(request.input_tokens) > 100 else "stop"
                }
                request.future.set_result(response)

    def _update_batch_stats(self, batch: Batch, processing_time: float):
        """Update batch processing statistics."""
        self.stats["batches_processed"] += 1

        # Update average batch size
        current_avg = self.stats["avg_batch_size"]
        total_batches = self.stats["batches_processed"]
        self.stats["avg_batch_size"] = \
            (current_avg * (total_batches - 1) + batch.get_batch_size()) / total_batches

        # Update average latency
        current_latency = self.stats["avg_latency_ms"]
        self.stats["avg_latency_ms"] = \
            (current_latency * (total_batches - 1) + processing_time) / total_batches

        # Calculate efficiency ratio (actual batch size / max batch size)
        efficiency = batch.get_batch_size() / self.max_batch_size
        current_efficiency = self.stats["efficiency_ratio"]
        self.stats["efficiency_ratio"] = \
            (current_efficiency * (total_batches - 1) + efficiency) / total_batches

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            **self.stats,
            "active_batches": len([b for b in self.active_batches.values() if b.requests]),
            "queued_requests": sum(len(queue) for queue in self.request_queues.values())
        }

    def get_gpu_queue_status(self, gpu_id: int) -> Dict[str, Any]:
        """Get queue status for a specific GPU."""
        if gpu_id not in self.request_queues:
            return {"error": f"GPU {gpu_id} not initialized"}

        queue = self.request_queues[gpu_id]
        batch = self.active_batches[gpu_id]

        return {
            "gpu_id": gpu_id,
            "queued_requests": len(queue),
            "active_batch_size": batch.get_batch_size(),
            "active_batch_memory": batch.get_memory_estimate(),
            "oldest_request_age": time.time() - min((r.timestamp for r in queue), default=time.time())
        }

    async def flush_all_batches(self):
        """Force processing of all pending batches."""
        logger.info("Flushing all pending batches...")

        for gpu_id, batch in self.active_batches.items():
            if batch.requests:
                await self._process_batch(batch)
                # Create new empty batch
                self.active_batches[gpu_id] = Batch(
                    batch_id=f"batch_{gpu_id}_{int(time.time())}",
                    gpu_id=gpu_id
                )

    def shutdown(self):
        """Shutdown the batch engine."""
        logger.info("Shutting down batch engine...")
        self._shutdown_event.set()

        # Cancel all pending futures
        for batch in self.active_batches.values():
            for request in batch.requests:
                if request.future and not request.future.done():
                    request.future.cancel()
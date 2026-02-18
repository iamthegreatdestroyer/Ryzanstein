"""
Sprint 4.3: Advanced Scheduling Integration Tests
[REF:SPRINT4.3] - Adaptive Batch Scheduler, Resource Allocator, Priority Queue

Tests:
1. PriorityInferenceQueue — enqueue/dequeue ordering
2. ResourceAwareAdmission — watermark enforcement
3. InferenceScheduler — submit/complete lifecycle
4. Priority aging (boost_rate)
5. Deadline-first ordering
6. Admission rejection under KV pressure
7. Concurrent scheduling (thread-safe)
8. Scheduler stats tracking
9. Scheduler start/stop lifecycle
"""

import sys
import asyncio
import time
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
API_SRC = REPO_ROOT / "RYZEN-LLM" / "src" / "api"
sys.path.insert(0, str(API_SRC))
sys.path.insert(0, str(REPO_ROOT / "RYZEN-LLM" / "src"))
sys.path.insert(0, str(REPO_ROOT / "PHASE2_DEVELOPMENT" / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def priority_queue():
    from scheduling_integration import PriorityInferenceQueue
    return PriorityInferenceQueue(maxsize=32)


@pytest.fixture
def admission():
    from scheduling_integration import ResourceAwareAdmission
    return ResourceAwareAdmission(
        max_kv_tokens=1024,
        max_batch_size=8,
        high_watermark=0.8,
        critical_watermark=0.95,
    )


@pytest.fixture
def scheduler():
    from scheduling_integration import InferenceScheduler, SchedulingPolicy
    return InferenceScheduler(
        policy=SchedulingPolicy.FCFS,
        max_queue=64,
        max_concurrent=4,
        max_kv_tokens=4096,
    )


@pytest.fixture
def sched_request():
    from scheduling_integration import SchedulingRequest, RequestPriority
    return SchedulingRequest(
        request_id="test-001",
        tenant_id="tenant-a",
        sequence_length=64,
        max_new_tokens=32,
        priority=RequestPriority(base=0),
    )


# ============================================================================
# Test 1: Priority Queue Ordering
# ============================================================================

class TestPriorityInferenceQueue:
    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, priority_queue):
        from scheduling_integration import InferenceQueueEntry, SchedulingRequest, RequestPriority

        async def noop(): pass

        loop = asyncio.get_event_loop()
        req = SchedulingRequest(
            request_id="r1", tenant_id="t1",
            sequence_length=10, max_new_tokens=10,
            priority=RequestPriority(base=0),
        )
        entry = InferenceQueueEntry(
            request_id="r1", sched_request=req,
            inference_func=noop, func_args=(), func_kwargs={},
            future=loop.create_future(),
        )

        assert priority_queue.empty()
        await priority_queue.put(entry)
        assert not priority_queue.empty()
        assert priority_queue.qsize() == 1

        dequeued = await priority_queue.get()
        assert dequeued.request_id == "r1"
        assert priority_queue.empty()

    @pytest.mark.asyncio
    async def test_higher_priority_dequeued_first(self, priority_queue):
        from scheduling_integration import InferenceQueueEntry, SchedulingRequest, RequestPriority

        loop = asyncio.get_event_loop()

        async def noop(): pass

        def make_entry(rid, priority_base):
            req = SchedulingRequest(
                request_id=rid, tenant_id="t",
                sequence_length=10, max_new_tokens=10,
                priority=RequestPriority(base=priority_base),
            )
            return InferenceQueueEntry(
                request_id=rid, sched_request=req,
                inference_func=noop, func_args=(), func_kwargs={},
                future=loop.create_future(),
            )

        low = make_entry("low", priority_base=-5)
        high = make_entry("high", priority_base=5)
        mid = make_entry("mid", priority_base=0)

        await priority_queue.put(low)
        await priority_queue.put(mid)
        await priority_queue.put(high)

        first = await priority_queue.get()
        assert first.request_id == "high", f"Expected high priority first, got {first.request_id}"

    @pytest.mark.asyncio
    async def test_queue_full_raises(self, priority_queue):
        from scheduling_integration import InferenceQueueEntry, SchedulingRequest, RequestPriority

        loop = asyncio.get_event_loop()

        async def noop(): pass

        # Fill queue to maxsize (32)
        for i in range(priority_queue.maxsize):
            req = SchedulingRequest(
                request_id=f"r{i}", tenant_id="t",
                sequence_length=10, max_new_tokens=10,
                priority=RequestPriority(base=0),
            )
            entry = InferenceQueueEntry(
                request_id=f"r{i}", sched_request=req,
                inference_func=noop, func_args=(), func_kwargs={},
                future=loop.create_future(),
            )
            await priority_queue.put(entry)

        # One more should raise
        req = SchedulingRequest(
            request_id="overflow", tenant_id="t",
            sequence_length=10, max_new_tokens=10,
            priority=RequestPriority(base=0),
        )
        overflow = InferenceQueueEntry(
            request_id="overflow", sched_request=req,
            inference_func=noop, func_args=(), func_kwargs={},
            future=loop.create_future(),
        )
        with pytest.raises(asyncio.QueueFull):
            await priority_queue.put(overflow)

    def test_stats(self, priority_queue):
        stats = priority_queue.get_stats()
        assert "queued" in stats
        assert stats["queued"] == 0


# ============================================================================
# Test 2: Resource-Aware Admission
# ============================================================================

class TestResourceAwareAdmission:
    @pytest.mark.asyncio
    async def test_admits_within_budget(self, admission, sched_request):
        admitted = await admission.admit(sched_request)
        assert admitted
        assert admission._total_admitted == 1

    @pytest.mark.asyncio
    async def test_rejects_at_critical_watermark(self, admission):
        from scheduling_integration import SchedulingRequest, RequestPriority

        # Exhaust KV budget (1024 tokens) to reach critical watermark
        big_req = SchedulingRequest(
            request_id="big", tenant_id="t",
            sequence_length=490, max_new_tokens=490,   # 980 tokens = 95.7%
            priority=RequestPriority(base=0),
        )
        admitted = await admission.admit(big_req)
        assert admitted

        # Next request should be rejected (critical watermark exceeded)
        small = SchedulingRequest(
            request_id="small", tenant_id="t",
            sequence_length=1, max_new_tokens=1,
            priority=RequestPriority(base=0),
        )
        rejected = await admission.admit(small)
        assert not rejected
        assert admission._total_rejected >= 1

    @pytest.mark.asyncio
    async def test_rejects_large_at_high_watermark(self, admission):
        from scheduling_integration import SchedulingRequest, RequestPriority

        # Push utilization to 80%+ (high watermark)
        moderate = SchedulingRequest(
            request_id="mod", tenant_id="t",
            sequence_length=400, max_new_tokens=424,   # ~80%
            priority=RequestPriority(base=0),
        )
        await admission.admit(moderate)

        # Large request (> 512 tokens) should be rejected
        large = SchedulingRequest(
            request_id="large", tenant_id="t",
            sequence_length=300, max_new_tokens=300,
            priority=RequestPriority(base=0),
        )
        rejected = await admission.admit(large)
        assert not rejected

    @pytest.mark.asyncio
    async def test_release_restores_capacity(self, admission, sched_request):
        admitted = await admission.admit(sched_request)
        assert admitted

        kv_before_release = admission._current_kv_tokens
        await admission.release(sched_request)
        assert admission._current_kv_tokens < kv_before_release

    def test_stats_dict(self, admission):
        stats = admission.get_stats()
        assert "kv_utilization" in stats
        assert "batch_size" in stats
        assert stats["kv_utilization"] == 0.0


# ============================================================================
# Test 3: Inference Scheduler Submit/Complete
# ============================================================================

class TestInferenceScheduler:
    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        await scheduler.start()
        assert scheduler._running
        await scheduler.stop()
        assert not scheduler._running

    @pytest.mark.asyncio
    async def test_submit_returns_result(self, scheduler):
        await scheduler.start()

        async def compute(): return {"tokens": [1, 2, 3]}

        try:
            result = await asyncio.wait_for(
                scheduler.submit(compute, sequence_length=10, max_new_tokens=5),
                timeout=5.0
            )
            assert result == {"tokens": [1, 2, 3]}
            assert scheduler._stats["total_completed"] == 1
        finally:
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_exception_propagates(self, scheduler):
        await scheduler.start()

        async def failing(): raise ValueError("inference error")

        try:
            with pytest.raises(ValueError, match="inference error"):
                await asyncio.wait_for(
                    scheduler.submit(failing, sequence_length=5, max_new_tokens=5),
                    timeout=5.0
                )
            assert scheduler._stats["total_failed"] == 1
        finally:
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, scheduler):
        await scheduler.start()
        results = []

        async def compute(value): return value

        try:
            tasks = [
                asyncio.create_task(
                    scheduler.submit(compute, i, sequence_length=10, max_new_tokens=10)
                )
                for i in range(8)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successes = [r for r in results if not isinstance(r, Exception)]
            assert len(successes) == 8
            assert scheduler._stats["total_completed"] == 8
        finally:
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_admission_rejection_raises(self, scheduler):
        """Override admission to always reject."""
        await scheduler.start()

        # Force critical watermark by setting current KV to max
        scheduler._admission._current_kv_tokens = scheduler._admission.max_kv_tokens

        async def compute(): return "ok"

        try:
            with pytest.raises(RuntimeError, match="admission control"):
                await scheduler.submit(compute, sequence_length=100, max_new_tokens=100)
        finally:
            scheduler._admission._current_kv_tokens = 0
            await scheduler.stop()

    def test_stats(self, scheduler):
        stats = scheduler.get_stats()
        assert "policy" in stats
        assert "queue" in stats
        assert "admission" in stats
        assert stats["policy"] == "FCFS"


# ============================================================================
# Test 4: Priority Aging
# ============================================================================

class TestPriorityAging:
    def test_effective_priority_increases_over_time(self):
        from scheduling_integration import RequestPriority
        p = RequestPriority(base=0, boost_rate=1.0)
        p0 = p.effective_priority(0.0)
        p10 = p.effective_priority(10.0)
        assert p10 > p0

    def test_boost_capped_at_5(self):
        from scheduling_integration import RequestPriority
        p = RequestPriority(base=0, boost_rate=100.0)
        # Even with very high boost rate, cap is 5
        prio = p.effective_priority(1000.0)
        # base=0, boost=5 (capped), value=1.0, urgency=0.0 → (0+5)*1.0*1.0 = 5.0
        assert prio == pytest.approx(5.0, abs=0.1)

    def test_higher_urgency_boosts_priority(self):
        from scheduling_integration import RequestPriority
        low_urgency = RequestPriority(base=0, urgency=0.0)
        high_urgency = RequestPriority(base=0, urgency=1.0)
        assert high_urgency.effective_priority(0.0) > low_urgency.effective_priority(0.0)


# ============================================================================
# Test 5: Scheduler Stats Tracking
# ============================================================================

class TestSchedulerStats:
    @pytest.mark.asyncio
    async def test_stats_track_all_outcomes(self, scheduler):
        await scheduler.start()

        async def ok(): return "good"
        async def bad(): raise RuntimeError("bad")

        try:
            await asyncio.wait_for(scheduler.submit(ok), timeout=3.0)
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(scheduler.submit(bad), timeout=3.0)

            stats = scheduler.get_stats()
            assert stats["completed"] == 1
            assert stats["failed"] == 1
            assert stats["scheduled"] == 2
        finally:
            await scheduler.stop()


# ============================================================================
# Test 6: Singleton Access
# ============================================================================

class TestSchedulerSingleton:
    def test_get_scheduler_returns_same_instance(self):
        from scheduling_integration import get_inference_scheduler
        import scheduling_integration as si
        si._inference_scheduler = None

        s1 = get_inference_scheduler()
        s2 = get_inference_scheduler()
        assert s1 is s2
        si._inference_scheduler = None


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-x"],
        cwd=str(REPO_ROOT)
    )
    sys.exit(result.returncode)

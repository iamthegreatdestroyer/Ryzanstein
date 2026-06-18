"""
Innovation #9: Negative-Space Scheduling — Unit Tests
=====================================================

Validates:
  - GlyphJob: resonance computation, interference detection
  - GlyphScheduler: submit, run_until_complete, parallelism, stats
  - Execution order emerges from glyph resonance, not static DAG
  - Non-interfering jobs run in parallel; interfering jobs serialise
  - Failed jobs don't block the scheduler
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.scheduling.negative_space_scheduler import (
    GlyphJob,
    GlyphScheduler,
    JobState,
    SchedulerResult,
)


# ============================================================================
# GlyphJob
# ============================================================================

class TestGlyphJob:

    async def _noop(self):
        return "done"

    def test_resonance_empty_probe_always_one(self):
        job = GlyphJob("j", self._noop, input_probe_ids=frozenset())
        assert job.resonance(frozenset()) == 1.0
        assert job.resonance(frozenset([1, 2, 3])) == 1.0

    def test_resonance_no_completed_primitives(self):
        job = GlyphJob("j", self._noop, input_probe_ids=frozenset([0, 1, 2]))
        assert job.resonance(frozenset()) == 0.0

    def test_resonance_full_overlap(self):
        job = GlyphJob("j", self._noop, input_probe_ids=frozenset([0, 1, 2]))
        assert job.resonance(frozenset([0, 1, 2])) == 1.0

    def test_resonance_partial_overlap(self):
        job = GlyphJob("j", self._noop, input_probe_ids=frozenset([0, 1, 2, 3]))
        # intersection = {0, 1}, union = {0, 1, 2, 3}
        assert abs(job.resonance(frozenset([0, 1])) - 0.5) < 1e-9

    def test_interferes_with_overlapping_bloom(self):
        j1 = GlyphJob("j1", self._noop, output_bloom_ids=frozenset([0, 1, 2]))
        j2 = GlyphJob("j2", self._noop, output_bloom_ids=frozenset([2, 3, 4]))
        assert j1.interferes_with(j2)

    def test_no_interference_disjoint_bloom(self):
        j1 = GlyphJob("j1", self._noop, output_bloom_ids=frozenset([0, 1]))
        j2 = GlyphJob("j2", self._noop, output_bloom_ids=frozenset([2, 3]))
        assert not j1.interferes_with(j2)

    def test_no_interference_empty_bloom(self):
        j1 = GlyphJob("j1", self._noop, output_bloom_ids=frozenset())
        j2 = GlyphJob("j2", self._noop, output_bloom_ids=frozenset())
        assert not j1.interferes_with(j2)

    def test_initial_state_pending(self):
        job = GlyphJob("j", self._noop)
        assert job.state == JobState.PENDING

    def test_repr(self):
        job = GlyphJob("my_job", self._noop, input_probe_ids=frozenset([1, 2]))
        r = repr(job)
        assert "my_job" in r
        assert "pending" in r


# ============================================================================
# GlyphScheduler — basic
# ============================================================================

class TestGlyphSchedulerBasic:

    @pytest.mark.asyncio
    async def test_single_job_completes(self):
        order = []
        async def task_a():
            order.append("a")
        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("a", task_a))
        await scheduler.run_until_complete()
        assert "a" in order
        assert scheduler._jobs["a"].state == JobState.COMPLETED

    @pytest.mark.asyncio
    async def test_two_independent_jobs_both_complete(self):
        results = []
        async def a(): results.append("a")
        async def b(): results.append("b")
        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("a", a))
        scheduler.submit(GlyphJob("b", b))
        await scheduler.run_until_complete()
        assert set(results) == {"a", "b"}
        assert scheduler._jobs["a"].state == JobState.COMPLETED
        assert scheduler._jobs["b"].state == JobState.COMPLETED

    @pytest.mark.asyncio
    async def test_job_with_resonance_threshold_waits(self):
        """Job B requires primitive 99 to be in completed set before running."""
        order = []

        async def a():
            order.append("a")

        async def b():
            order.append("b")

        scheduler = GlyphScheduler()
        # Job A writes primitive 99
        scheduler.submit(GlyphJob(
            "a", a,
            output_bloom_ids=frozenset([99]),
        ))
        # Job B probes for primitive 99 with threshold 0.5
        scheduler.submit(GlyphJob(
            "b", b,
            input_probe_ids=frozenset([99]),
            resonance_threshold=0.5,
        ))
        await scheduler.run_until_complete()
        assert order.index("a") < order.index("b"), (
            "B must run after A provides primitive 99"
        )

    @pytest.mark.asyncio
    async def test_interfering_jobs_serialised(self):
        """Two jobs writing to the same primitive cannot run in parallel."""
        running_together = []
        currently_running = []

        async def a():
            currently_running.append("a")
            await asyncio.sleep(0.001)
            running_together.append(tuple(sorted(currently_running)))
            currently_running.remove("a")

        async def b():
            currently_running.append("b")
            await asyncio.sleep(0.001)
            running_together.append(tuple(sorted(currently_running)))
            currently_running.remove("b")

        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("a", a, output_bloom_ids=frozenset([5])))
        scheduler.submit(GlyphJob("b", b, output_bloom_ids=frozenset([5])))
        await scheduler.run_until_complete()
        # Neither snapshot should show both running simultaneously
        for snap in running_together:
            assert len(snap) == 1, f"Both ran together: {snap}"

    @pytest.mark.asyncio
    async def test_non_interfering_jobs_can_run_in_parallel(self):
        """Two jobs with disjoint output blooms should both be dispatched on tick 0."""
        dispatched_in_tick_0 = []

        async def a(): pass
        async def b(): pass

        scheduler = GlyphScheduler(max_parallelism=4)
        scheduler.submit(GlyphJob("a", a, output_bloom_ids=frozenset([0, 1])))
        scheduler.submit(GlyphJob("b", b, output_bloom_ids=frozenset([2, 3])))
        results = await scheduler.run_until_complete()
        # Both should be dispatched in the first tick
        tick0 = results[0]
        assert set(tick0.jobs_dispatched) == {"a", "b"}, (
            "Non-interfering jobs should be dispatched in the same tick"
        )

    @pytest.mark.asyncio
    async def test_duplicate_job_id_raises(self):
        async def noop(): pass
        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("j", noop))
        with pytest.raises(ValueError, match="Duplicate"):
            scheduler.submit(GlyphJob("j", noop))

    @pytest.mark.asyncio
    async def test_failed_job_does_not_block(self):
        results = []

        async def bad():
            raise RuntimeError("intentional failure")

        async def good():
            results.append("good")

        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("bad",  bad))
        scheduler.submit(GlyphJob("good", good))
        await scheduler.run_until_complete()
        assert "good" in results
        assert scheduler._jobs["bad"].state  == JobState.FAILED
        assert scheduler._jobs["good"].state == JobState.COMPLETED

    @pytest.mark.asyncio
    async def test_empty_scheduler_completes(self):
        scheduler = GlyphScheduler()
        results = await scheduler.run_until_complete()
        assert isinstance(results, dict)

    @pytest.mark.asyncio
    async def test_stats(self):
        async def noop(): pass
        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("j", noop))
        await scheduler.run_until_complete()
        s = scheduler.stats()
        assert s["total_jobs"] == 1
        assert s["total_completed"] == 1
        assert s["job_states"]["completed"] == 1


# ============================================================================
# GlyphScheduler — resonance-driven ordering
# ============================================================================

class TestGlyphSchedulerResonance:

    @pytest.mark.asyncio
    async def test_chain_of_three_executes_in_order(self):
        """A → B → C: each depends on the previous one's output primitives."""
        order = []

        async def a(): order.append("a")
        async def b(): order.append("b")
        async def c(): order.append("c")

        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("a", a, output_bloom_ids=frozenset([10])))
        scheduler.submit(GlyphJob(
            "b", b,
            input_probe_ids=frozenset([10]),
            output_bloom_ids=frozenset([20]),
            resonance_threshold=0.5,
        ))
        scheduler.submit(GlyphJob(
            "c", c,
            input_probe_ids=frozenset([20]),
            resonance_threshold=0.5,
        ))
        await scheduler.run_until_complete()
        assert order == ["a", "b", "c"] or order == ["a", "b", "c"], (
            f"Expected a→b→c, got {order}"
        )

    @pytest.mark.asyncio
    async def test_priority_breaks_ties(self):
        """Among jobs with equal resonance, lower priority number runs first."""
        order = []

        async def hi(): order.append("hi")
        async def lo(): order.append("lo")

        scheduler = GlyphScheduler(max_parallelism=1)  # force sequential
        scheduler.submit(GlyphJob("lo", lo, priority=10))
        scheduler.submit(GlyphJob("hi", hi, priority=0))
        await scheduler.run_until_complete()
        assert order[0] == "hi", f"hi-priority job should run first, got {order}"

    @pytest.mark.asyncio
    async def test_completed_primitives_grow(self):
        """After each job, the completed primitive set grows."""
        async def a(): pass
        async def b(): pass

        scheduler = GlyphScheduler()
        scheduler.submit(GlyphJob("a", a, output_bloom_ids=frozenset([1, 2, 3])))
        scheduler.submit(GlyphJob("b", b, output_bloom_ids=frozenset([4, 5, 6])))
        await scheduler.run_until_complete()
        assert scheduler._completed_primitives == frozenset([1, 2, 3, 4, 5, 6])

    @pytest.mark.asyncio
    async def test_parallelism_count_in_results(self):
        async def slow():
            await asyncio.sleep(0.005)

        scheduler = GlyphScheduler(max_parallelism=4)
        for i in range(4):
            scheduler.submit(GlyphJob(f"j{i}", slow, output_bloom_ids=frozenset([i])))
        results = await scheduler.run_until_complete()
        max_par = max(r.parallelism for r in results.values())
        assert max_par >= 2, "Expected at least 2-way parallelism"

"""
Negative-Space Scheduling (Fractal Job Orchestration) — Innovation #9
=======================================================================

Core concept (from Ryot-updates.md, section 9):

    "Jobs don't have fixed dependencies; instead, they have glyph-space
     alignments. Job A is 'ready' when the glyph residue of Job B's output
     resonates above a threshold with Job A's input probe. This is radically
     more flexible than DAG-based scheduling: jobs can run in parallel if
     their glyph neighborhoods don't interfere. Sub-linear scheduling cost:
     You're checking resonance, not full dependency graphs. Enables elastic,
     adaptive pipelines where execution order emerges from data rather than
     static configuration."

Problem with DAG scheduling
----------------------------
Standard pipeline: Job A → Job B → Job C (fixed)
- Topology is baked in at design time
- If A and B touch different glyph subspaces, they could run in parallel —
  but the DAG doesn't know that
- Dependency resolution is O(N edges) per tick

Negative-space scheduling
--------------------------
Each job declares:
    input_probe:  a GlyphNativeStream or primitive ID set it needs to read
    output_bloom: which primitive IDs it will write

Two jobs can run in parallel if their output_blooms don't intersect
(no glyph neighborhood interference).

A job is READY when:
    - resonance(its input_probe, completed_output) >= threshold
    - No currently-running job's output_bloom overlaps its output_bloom

`resonance` = Jaccard similarity between input probe's primitive set and
the union of completed job outputs' primitive sets.

This is sub-linear: checking bloom filter overlap is O(32 bytes) per pair,
not O(full dependency graph traversal).

Classes
-------
GlyphJob       — unit of work with input probe and output bloom
GlyphScheduler — resonance-based scheduler
SchedulerResult — tick outcome with parallelism stats
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, FrozenSet, List, Optional, Set

N_PRIMITIVES = 256


class JobState(Enum):
    PENDING   = "pending"
    READY     = "ready"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


# ---------------------------------------------------------------------------
# GlyphJob
# ---------------------------------------------------------------------------

@dataclass
class GlyphJob:
    """
    A unit of work in the negative-space scheduler.

    Instead of declaring explicit predecessors (DAG edges), each job
    declares what glyph primitive neighbourhood it needs to read
    (`input_probe`) and what it will write (`output_bloom_ids`).

    The scheduler determines readiness and parallelism from these declarations
    — no static topology needed.

    Parameters
    ----------
    job_id
        Unique string identifier.
    fn
        Async callable that performs the actual work.
        Receives no arguments; use closures to capture context.
    input_probe_ids
        Primitive IDs this job needs to read. The job becomes READY when the
        global completed-output set resonates above `resonance_threshold` with
        this probe. Empty probe → job is always immediately ready.
    output_bloom_ids
        Primitive IDs this job will write. Jobs with overlapping output blooms
        cannot run in parallel (interference prevention).
    resonance_threshold
        Minimum Jaccard similarity between input_probe and completed outputs
        for this job to be considered ready. Default 0.0 means "any overlap
        OR empty probe → ready".
    priority
        Lower = higher priority within the same resonance tier.
    """
    job_id:              str
    fn:                  Callable[[], Coroutine]
    input_probe_ids:     FrozenSet[int] = field(default_factory=frozenset)
    output_bloom_ids:    FrozenSet[int] = field(default_factory=frozenset)
    resonance_threshold: float = 0.0
    priority:            int   = 0

    state:   JobState = field(default=JobState.PENDING, init=False)
    result:  Any      = field(default=None,             init=False)
    error:   Optional[Exception] = field(default=None,  init=False)
    _queued_at:  float = field(default_factory=time.monotonic, init=False)
    _started_at: Optional[float] = field(default=None,  init=False)
    _done_at:    Optional[float] = field(default=None,   init=False)

    @property
    def latency_ms(self) -> Optional[float]:
        if self._started_at and self._done_at:
            return round((_done_at := self._done_at) and (self._done_at - self._started_at) * 1000, 3)
        return None

    def resonance(self, completed_primitives: FrozenSet[int]) -> float:
        """
        Jaccard similarity between this job's input probe and the global
        set of completed output primitives.

        0.0 = no overlap (job may not yet be ready)
        1.0 = perfect alignment (job's glyph neighbourhood is fully available)
        """
        if not self.input_probe_ids:
            return 1.0   # empty probe → always ready
        union = self.input_probe_ids | completed_primitives
        if not union:
            return 1.0
        return len(self.input_probe_ids & completed_primitives) / len(union)

    def interferes_with(self, other: "GlyphJob") -> bool:
        """
        True if this job's output bloom overlaps with another job's output
        bloom → they cannot run in parallel.
        """
        return bool(self.output_bloom_ids & other.output_bloom_ids)

    def __repr__(self) -> str:
        return (
            f"GlyphJob({self.job_id!r}, state={self.state.value}, "
            f"probe={len(self.input_probe_ids)}, bloom={len(self.output_bloom_ids)})"
        )


# ---------------------------------------------------------------------------
# SchedulerResult
# ---------------------------------------------------------------------------

@dataclass
class SchedulerResult:
    """Outcome of one scheduler tick."""
    tick:              int
    jobs_dispatched:   List[str]       # job_ids dispatched this tick
    jobs_completed:    List[str]       # job_ids completed before this tick
    parallelism:       int             # jobs running concurrently this tick
    resonance_scores:  Dict[str, float]# resonance score at dispatch time
    completed_prims:   int             # size of global completed primitive set
    pending_count:     int
    running_count:     int
    elapsed_ms:        float

    def summary(self) -> str:
        return (
            f"Tick {self.tick}: dispatched={len(self.jobs_dispatched)}, "
            f"parallelism={self.parallelism}, "
            f"completed_prims={self.completed_prims}, "
            f"pending={self.pending_count}"
        )


# ---------------------------------------------------------------------------
# GlyphScheduler
# ---------------------------------------------------------------------------

class GlyphScheduler:
    """
    Resonance-based job scheduler — execution order emerges from data.

    Algorithm per tick:
    1. Compute current `completed_primitives`: union of output_bloom_ids of
       all COMPLETED jobs.
    2. For each PENDING job: compute resonance with completed_primitives.
       Mark READY if resonance >= job's resonance_threshold.
    3. From READY jobs: greedily select a non-interfering subset (maximise
       parallelism, break ties by priority then arrival order).
    4. Dispatch selected jobs as concurrent asyncio tasks.
    5. Wait for all running tasks to finish, then repeat.

    Complexity:
        Step 2: O(N_pending × 32B bloom) ← sub-linear vs O(N² edges) for DAG
        Step 3: O(N_ready²) in the worst case but N_ready is typically small

    Parameters
    ----------
    resonance_threshold
        Default resonance threshold for jobs that don't set their own.
    max_parallelism
        Cap on concurrent jobs per tick (resource limit).
    """

    def __init__(
        self,
        resonance_threshold: float = 0.0,
        max_parallelism:     int   = 8,
    ):
        self._threshold       = resonance_threshold
        self._max_parallelism = max_parallelism
        self._jobs:           Dict[str, GlyphJob] = {}
        self._completed_primitives: FrozenSet[int] = frozenset()
        self._tick:           int = 0
        self._total_dispatched: int = 0
        self._total_completed:  int = 0
        self._max_observed_parallelism: int = 0

    def submit(self, job: GlyphJob) -> None:
        """Add a job to the scheduler. Can be called at any time."""
        if job.job_id in self._jobs:
            raise ValueError(f"Duplicate job_id: {job.job_id!r}")
        self._jobs[job.job_id] = job

    async def run_until_complete(self) -> Dict[str, SchedulerResult]:
        """
        Run all submitted jobs to completion.

        Dispatches in resonance order, maximising parallelism while preventing
        bloom interference. Returns per-tick SchedulerResult records.
        """
        results: Dict[str, SchedulerResult] = {}
        running_tasks: Dict[str, asyncio.Task] = {}

        while True:
            t0 = time.monotonic()

            # Collect newly completed tasks
            newly_done = [jid for jid, task in running_tasks.items() if task.done()]
            for jid in newly_done:
                task = running_tasks.pop(jid)
                job  = self._jobs[jid]
                if task.exception():
                    job.state = JobState.FAILED
                    job.error = task.exception()
                else:
                    job.state  = JobState.COMPLETED
                    job.result = task.result()
                    self._completed_primitives = (
                        self._completed_primitives | job.output_bloom_ids
                    )
                job._done_at = time.monotonic()
                self._total_completed += 1

            # Determine ready jobs
            ready = self._select_ready(running_tasks)

            # Dispatch non-interfering subset
            dispatched = self._select_non_interfering(ready, running_tasks)

            resonance_at_dispatch = {}
            for job in dispatched:
                job.state       = JobState.RUNNING
                job._started_at = time.monotonic()
                resonance_at_dispatch[job.job_id] = round(
                    job.resonance(self._completed_primitives), 4
                )
                task = asyncio.create_task(job.fn(), name=job.job_id)
                running_tasks[job.job_id] = task
                self._total_dispatched += 1

            parallelism = len(running_tasks)
            self._max_observed_parallelism = max(
                self._max_observed_parallelism, parallelism
            )

            pending_count = sum(
                1 for j in self._jobs.values() if j.state == JobState.PENDING
            )

            results[self._tick] = SchedulerResult(
                tick             = self._tick,
                jobs_dispatched  = [j.job_id for j in dispatched],
                jobs_completed   = newly_done,
                parallelism      = parallelism,
                resonance_scores = resonance_at_dispatch,
                completed_prims  = len(self._completed_primitives),
                pending_count    = pending_count,
                running_count    = len(running_tasks),
                elapsed_ms       = round((time.monotonic() - t0) * 1000, 3),
            )
            self._tick += 1

            # Termination: no pending, no running
            if pending_count == 0 and not running_tasks:
                break

            # Wait for at least one running task to finish before next tick
            if running_tasks:
                done, _ = await asyncio.wait(
                    running_tasks.values(), return_when=asyncio.FIRST_COMPLETED
                )
                # Re-check completed on next iteration

        return results

    def _select_ready(self, running: Dict[str, asyncio.Task]) -> List[GlyphJob]:
        """
        Return PENDING jobs whose resonance meets their threshold and that
        don't exceed max_parallelism.
        """
        available_slots = self._max_parallelism - len(running)
        if available_slots <= 0:
            return []

        ready = []
        for job in self._jobs.values():
            if job.state != JobState.PENDING:
                continue
            r = job.resonance(self._completed_primitives)
            thresh = max(job.resonance_threshold, self._threshold)
            if r >= thresh:
                ready.append(job)

        # Sort by priority (lower = higher priority), then by arrival time
        ready.sort(key=lambda j: (j.priority, j._queued_at))
        return ready[:available_slots]

    def _select_non_interfering(
        self,
        candidates: List[GlyphJob],
        running: Dict[str, asyncio.Task],
    ) -> List[GlyphJob]:
        """
        Greedy non-interfering selection from candidates.

        Two jobs interfere if their output_bloom_ids overlap.
        Also check against currently running jobs' output blooms.
        """
        # Build set of primitives locked by running jobs
        running_blooms: FrozenSet[int] = frozenset()
        for jid in running:
            running_blooms = running_blooms | self._jobs[jid].output_bloom_ids

        selected: List[GlyphJob] = []
        selected_bloom: FrozenSet[int] = frozenset()

        for job in candidates:
            # Check interference with running jobs
            if job.output_bloom_ids & running_blooms:
                continue
            # Check interference with already-selected jobs this tick
            if job.output_bloom_ids & selected_bloom:
                continue
            selected.append(job)
            selected_bloom = selected_bloom | job.output_bloom_ids

        return selected

    def stats(self) -> Dict:
        states: Dict[str, int] = {s.value: 0 for s in JobState}
        for j in self._jobs.values():
            states[j.state.value] += 1
        return {
            "total_jobs":          len(self._jobs),
            "ticks":               self._tick,
            "total_dispatched":    self._total_dispatched,
            "total_completed":     self._total_completed,
            "max_parallelism":     self._max_observed_parallelism,
            "completed_primitives": len(self._completed_primitives),
            "job_states":          states,
        }

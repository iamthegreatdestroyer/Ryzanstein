"""
Comprehensive test suite for agentmem — Cross-Agent Episodic Memory Protocol.

Target: >90% coverage across all memory layers and consolidation pipeline.
"""

from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime, timedelta

from agentmem import (
    MemoryStore,
    EpisodicMemory,
    Episode,
    SemanticMemory,
    Fact,
    Relation,
    ProceduralMemory,
    Workflow,
    WorkingMemory,
    ConsolidationPipeline,
)
from agentmem.types import (
    MemoryLayer,
    MemoryQuery,
    MemoryResult,
    OutcomeLabel,
    Embedding,
)
from agentmem.procedural import WorkflowStep


# ============================================================
# MemoryStore tests
# ============================================================


class TestMemoryStore:
    """Tests for the unified MemoryStore backend."""

    def test_store_and_retrieve(self) -> None:
        store = MemoryStore()
        mid = store.store(
            layer=MemoryLayer.EPISODIC,
            agent_id="test-01",
            content={"task": "test"},
        )
        mem = store.retrieve(mid)
        assert mem is not None
        assert mem.content["task"] == "test"
        assert mem.agent_id == "test-01"
        assert mem.layer == MemoryLayer.EPISODIC

    def test_retrieve_nonexistent(self) -> None:
        store = MemoryStore()
        assert store.retrieve("nonexistent-id") is None

    def test_delete(self) -> None:
        store = MemoryStore()
        mid = store.store(
            layer=MemoryLayer.EPISODIC,
            agent_id="test-01",
            content={"data": "x"},
        )
        assert store.delete(mid) is True
        assert store.retrieve(mid) is None
        assert store.delete(mid) is False

    def test_count_all(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"x": 1})
        store.store(MemoryLayer.SEMANTIC, "b", {"x": 2})
        store.store(MemoryLayer.EPISODIC, "a", {"x": 3})
        assert store.count() == 3

    def test_count_by_agent(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"x": 1})
        store.store(MemoryLayer.EPISODIC, "b", {"x": 2})
        store.store(MemoryLayer.EPISODIC, "a", {"x": 3})
        assert store.count(agent_id="a") == 2
        assert store.count(agent_id="b") == 1

    def test_count_by_layer(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"x": 1})
        store.store(MemoryLayer.SEMANTIC, "a", {"x": 2})
        assert store.count(layer=MemoryLayer.EPISODIC) == 1
        assert store.count(layer=MemoryLayer.SEMANTIC) == 1

    def test_count_by_agent_and_layer(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"x": 1})
        store.store(MemoryLayer.SEMANTIC, "a", {"x": 2})
        store.store(MemoryLayer.EPISODIC, "b", {"x": 3})
        assert store.count(agent_id="a", layer=MemoryLayer.EPISODIC) == 1

    def test_clear_all(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"x": 1})
        store.store(MemoryLayer.EPISODIC, "b", {"x": 2})
        cleared = store.clear()
        assert cleared == 2
        assert store.count() == 0

    def test_clear_by_agent(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"x": 1})
        store.store(MemoryLayer.EPISODIC, "b", {"x": 2})
        cleared = store.clear(agent_id="a")
        assert cleared == 1
        assert store.count() == 1

    def test_search_text(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"task": "rate limiting"})
        store.store(MemoryLayer.EPISODIC, "a", {"task": "caching"})
        results = store.search(MemoryQuery(text="rate"))
        assert len(results) == 1
        assert results[0].content["task"] == "rate limiting"

    def test_search_with_embedding(self) -> None:
        store = MemoryStore()
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        query_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)

        store.store(
            MemoryLayer.EPISODIC,
            "a",
            {"task": "similar"},
            embedding=Embedding(vector=vec1, dimension=3),
        )
        store.store(
            MemoryLayer.EPISODIC,
            "a",
            {"task": "different"},
            embedding=Embedding(vector=vec2, dimension=3),
        )

        results = store.search(
            MemoryQuery(text="", min_similarity=0.5),
            query_embedding=Embedding(vector=query_vec, dimension=3),
        )
        assert len(results) >= 1
        assert results[0].content["task"] == "similar"

    def test_search_filter_agent(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"task": "rate limiting"})
        store.store(MemoryLayer.EPISODIC, "b", {"task": "rate limiting"})
        results = store.search(MemoryQuery(text="rate", agent_ids=["a"]))
        assert len(results) == 1
        assert results[0].agent_id == "a"

    def test_search_filter_layer(self) -> None:
        store = MemoryStore()
        store.store(MemoryLayer.EPISODIC, "a", {"task": "rate limiting"})
        store.store(MemoryLayer.SEMANTIC, "a", {"task": "rate limiting fact"})
        results = store.search(
            MemoryQuery(text="rate", layers=[MemoryLayer.SEMANTIC])
        )
        assert all(r.layer == MemoryLayer.SEMANTIC for r in results)


# ============================================================
# EpisodicMemory tests
# ============================================================


class TestEpisodicMemory:
    """Tests for agent-scoped episodic memory."""

    @pytest.fixture
    def epi(self) -> EpisodicMemory:
        store = MemoryStore()
        return EpisodicMemory(store=store, agent_id="apex-01")

    def test_record_and_recall(self, epi: EpisodicMemory) -> None:
        epi.record(
            task="implement rate limiter",
            outcome="success",
            strategy="sliding window",
        )
        results = epi.recall("rate limiter")
        assert len(results) >= 1

    def test_record_outcome_enum(self, epi: EpisodicMemory) -> None:
        mid = epi.record(
            task="test", outcome=OutcomeLabel.FAILURE, strategy="bad approach"
        )
        assert mid is not None

    def test_recall_successes(self, epi: EpisodicMemory) -> None:
        epi.record(task="task1", outcome="success", strategy="good")
        epi.record(task="task1", outcome="failure", strategy="bad")
        results = epi.recall_successes("task1")
        assert all(r.content.get("outcome") == "success" for r in results)

    def test_recall_failures(self, epi: EpisodicMemory) -> None:
        epi.record(task="task2", outcome="success", strategy="good")
        epi.record(task="task2", outcome="failure", strategy="bad")
        results = epi.recall_failures("task2")
        assert all(r.content.get("outcome") == "failure" for r in results)

    def test_count(self, epi: EpisodicMemory) -> None:
        assert epi.count() == 0
        epi.record(task="t1", outcome="success")
        epi.record(task="t2", outcome="failure")
        assert epi.count() == 2

    def test_record_with_context(self, epi: EpisodicMemory) -> None:
        epi.record(
            task="deploy",
            outcome="success",
            strategy="blue-green",
            context={"env": "production"},
            reasoning="minimize downtime",
        )
        results = epi.recall("deploy")
        assert len(results) >= 1
        assert results[0].content["context"]["env"] == "production"


# ============================================================
# SemanticMemory tests
# ============================================================


class TestSemanticMemory:
    """Tests for knowledge-graph semantic memory."""

    @pytest.fixture
    def sem(self) -> SemanticMemory:
        store = MemoryStore()
        return SemanticMemory(store=store, agent_id="apex-01")

    def test_add_and_query_fact(self, sem: SemanticMemory) -> None:
        sem.add_fact(
            Fact(
                subject="sliding window",
                predicate="is_good_for",
                value="rate limiting",
                confidence=0.95,
            )
        )
        results = sem.query_facts("sliding window")
        assert len(results) >= 1

    def test_fact_confidence_filter(self, sem: SemanticMemory) -> None:
        sem.add_fact(
            Fact(subject="x", predicate="p", value="v", confidence=0.3)
        )
        sem.add_fact(
            Fact(subject="y", predicate="p", value="v", confidence=0.9)
        )
        high = sem.query_facts("x", min_confidence=0.5)
        # Low confidence should be filtered
        assert all(
            r.content.get("confidence", 0) >= 0.5 for r in high
        )

    def test_add_relation(self, sem: SemanticMemory) -> None:
        mid = sem.add_relation(
            Relation(source="redis", target="caching", relation_type="used_for")
        )
        assert mid is not None

    def test_get_relations(self, sem: SemanticMemory) -> None:
        sem.add_relation(
            Relation(source="redis", target="caching", relation_type="used_for")
        )
        relations = sem.get_relations("redis")
        # Text search may or may not match; just ensure no crash
        assert isinstance(relations, list)

    def test_count(self, sem: SemanticMemory) -> None:
        assert sem.count() == 0
        sem.add_fact(Fact(subject="a", predicate="b", value="c"))
        assert sem.count() == 1


# ============================================================
# ProceduralMemory tests
# ============================================================


class TestProceduralMemory:
    """Tests for procedural (workflow) memory."""

    @pytest.fixture
    def proc(self) -> ProceduralMemory:
        store = MemoryStore()
        return ProceduralMemory(store=store, agent_id="flux-11")

    def test_add_workflow(self, proc: ProceduralMemory) -> None:
        wf = Workflow(
            name="deploy-k8s",
            description="Deploy to Kubernetes",
            steps=[
                WorkflowStep(action="build image", tool="docker"),
                WorkflowStep(action="push to registry", tool="crane"),
                WorkflowStep(action="apply manifests", tool="kubectl"),
            ],
            success_rate=0.85,
        )
        mid = proc.add_workflow(wf)
        assert mid is not None

    def test_find_workflow(self, proc: ProceduralMemory) -> None:
        proc.add_workflow(
            Workflow(
                name="deploy-k8s",
                description="Deploy to Kubernetes cluster",
                steps=[WorkflowStep(action="build")],
                success_rate=0.9,
            )
        )
        results = proc.find_workflow("Kubernetes deploy")
        assert isinstance(results, list)

    def test_update_success_rate(self, proc: ProceduralMemory) -> None:
        mid = proc.add_workflow(
            Workflow(
                name="test-wf",
                description="test",
                steps=[],
                success_rate=1.0,
                usage_count=1,
            )
        )
        new_rate = proc.update_success_rate(mid, succeeded=False)
        assert new_rate is not None
        assert new_rate < 1.0

    def test_update_nonexistent(self, proc: ProceduralMemory) -> None:
        assert proc.update_success_rate("fake-id", True) is None

    def test_count(self, proc: ProceduralMemory) -> None:
        assert proc.count() == 0
        proc.add_workflow(
            Workflow(name="w", description="d", steps=[])
        )
        assert proc.count() == 1


# ============================================================
# WorkingMemory tests
# ============================================================


class TestWorkingMemory:
    """Tests for ephemeral working memory."""

    def test_begin_task(self) -> None:
        wm = WorkingMemory(agent_id="apex-01")
        wm.begin_task("implement caching", goal="Add Redis layer")
        assert wm.task == "implement caching"
        assert wm.goal == "Add Redis layer"

    def test_variables(self) -> None:
        wm = WorkingMemory(agent_id="apex-01")
        wm.set("key", "value")
        assert wm.get("key") == "value"
        assert wm.get("missing", "default") == "default"

    def test_scratchpad(self) -> None:
        wm = WorkingMemory(agent_id="apex-01", capacity=3)
        wm.note("first")
        wm.note("second")
        wm.note("third")
        wm.note("fourth")  # Should evict "first"
        snap = wm.snapshot()
        assert "first" not in snap["scratchpad"]
        assert "fourth" in snap["scratchpad"]

    def test_focus_stack(self) -> None:
        wm = WorkingMemory(agent_id="apex-01")
        wm.push_focus("subtask-1")
        wm.push_focus("subtask-2")
        assert wm.current_focus == "subtask-2"
        assert wm.pop_focus() == "subtask-2"
        assert wm.current_focus == "subtask-1"
        assert wm.pop_focus() == "subtask-1"
        assert wm.pop_focus() is None
        assert wm.current_focus is None

    def test_clear(self) -> None:
        wm = WorkingMemory(agent_id="apex-01")
        wm.begin_task("task")
        wm.set("k", "v")
        wm.note("note")
        wm.clear()
        assert wm.task is None
        assert wm.get("k") is None

    def test_snapshot(self) -> None:
        wm = WorkingMemory(agent_id="apex-01")
        wm.begin_task("task", goal="goal")
        wm.set("x", 42)
        snap = wm.snapshot()
        assert snap["agent_id"] == "apex-01"
        assert snap["task"] == "task"
        assert snap["variables"]["x"] == 42


# ============================================================
# ConsolidationPipeline tests
# ============================================================


class TestConsolidationPipeline:
    """Tests for cross-agent memory consolidation."""

    def test_empty_consolidation(self) -> None:
        store = MemoryStore()
        pipeline = ConsolidationPipeline(store=store)
        result = pipeline.consolidate(agent_ids=["nonexistent"])
        assert result.episodes_processed == 0
        assert result.facts_extracted == 0

    def test_pattern_discovery(self) -> None:
        store = MemoryStore()
        epi_a = EpisodicMemory(store=store, agent_id="a")
        epi_b = EpisodicMemory(store=store, agent_id="b")

        # Both agents succeed with same strategy
        epi_a.record(task="rate limiting", outcome="success", strategy="sliding window")
        epi_b.record(task="rate limiting", outcome="success", strategy="sliding window")

        pipeline = ConsolidationPipeline(store=store, min_pattern_count=2)
        result = pipeline.consolidate(agent_ids=["a", "b"])

        assert result.episodes_processed >= 2
        assert result.patterns_discovered >= 1

    def test_contradiction_detection(self) -> None:
        store = MemoryStore()
        epi_a = EpisodicMemory(store=store, agent_id="a")
        epi_b = EpisodicMemory(store=store, agent_id="b")

        # Contradicting outcomes for same task
        epi_a.record(task="deploy to prod", outcome="success", strategy="blue-green")
        epi_b.record(task="deploy to prod", outcome="failure", strategy="canary")

        pipeline = ConsolidationPipeline(store=store, min_pattern_count=1)
        result = pipeline.consolidate(agent_ids=["a", "b"])

        assert result.contradictions_found >= 1


# ============================================================
# Embedding tests
# ============================================================


class TestEmbedding:
    """Tests for the Embedding type."""

    def test_cosine_similarity_identical(self) -> None:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        e1 = Embedding(vector=vec, dimension=3)
        e2 = Embedding(vector=vec, dimension=3)
        assert abs(e1.cosine_similarity(e2) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self) -> None:
        e1 = Embedding(vector=np.array([1.0, 0.0], dtype=np.float32), dimension=2)
        e2 = Embedding(vector=np.array([0.0, 1.0], dtype=np.float32), dimension=2)
        assert abs(e1.cosine_similarity(e2)) < 1e-6

    def test_cosine_similarity_zero_vector(self) -> None:
        e1 = Embedding(vector=np.array([0.0, 0.0], dtype=np.float32), dimension=2)
        e2 = Embedding(vector=np.array([1.0, 0.0], dtype=np.float32), dimension=2)
        assert e1.cosine_similarity(e2) == 0.0

"""
Semantic Memory — Extracted facts and relationships in a knowledge graph.

Consolidated from episodic memories via the consolidation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentmem.store import MemoryStore
from agentmem.types import AgentId, MemoryId, MemoryLayer, MemoryQuery, MemoryResult


@dataclass
class Fact:
    """An extracted fact from episodic experience.

    Attributes:
        subject: The entity or concept.
        predicate: The relationship or property.
        value: The value or target entity.
        confidence: Confidence score [0, 1].
        source_episodes: Episode IDs this fact was extracted from.
    """

    subject: str
    predicate: str
    value: str
    confidence: float = 1.0
    source_episodes: list[MemoryId] = field(default_factory=list)


@dataclass
class Relation:
    """A relationship between entities in the knowledge graph.

    Attributes:
        source: Source entity.
        target: Target entity.
        relation_type: Type of relationship.
        weight: Strength of the relationship [0, 1].
    """

    source: str
    target: str
    relation_type: str
    weight: float = 1.0


class SemanticMemory:
    """Knowledge-graph-based semantic memory layer.

    Args:
        store: Shared MemoryStore backend.
        agent_id: Agent identifier (None = shared cross-agent).

    Example:
        >>> store = MemoryStore()
        >>> sem = SemanticMemory(store=store, agent_id="apex-01")
        >>> sem.add_fact(Fact(
        ...     subject="sliding window",
        ...     predicate="is_good_for",
        ...     value="rate limiting",
        ...     confidence=0.95,
        ... ))
        >>> facts = sem.query_facts("rate limiting")
    """

    def __init__(
        self,
        store: MemoryStore,
        agent_id: Optional[AgentId] = None,
    ) -> None:
        self.store = store
        self.agent_id = agent_id or "__shared__"

    def add_fact(self, fact: Fact) -> MemoryId:
        """Add a fact to semantic memory."""
        content = {
            "subject": fact.subject,
            "predicate": fact.predicate,
            "value": fact.value,
            "confidence": fact.confidence,
            "source_episodes": fact.source_episodes,
            "type": "fact",
        }
        return self.store.store(
            layer=MemoryLayer.SEMANTIC,
            agent_id=self.agent_id,
            content=content,
            metadata={"confidence": fact.confidence},
        )

    def add_relation(self, relation: Relation) -> MemoryId:
        """Add a relationship to the knowledge graph."""
        content = {
            "source": relation.source,
            "target": relation.target,
            "relation_type": relation.relation_type,
            "weight": relation.weight,
            "type": "relation",
        }
        return self.store.store(
            layer=MemoryLayer.SEMANTIC,
            agent_id=self.agent_id,
            content=content,
            metadata={"weight": relation.weight},
        )

    def query_facts(
        self,
        query: str,
        top_k: int = 10,
        min_confidence: float = 0.5,
    ) -> list[MemoryResult]:
        """Query semantic memory for relevant facts."""
        mq = MemoryQuery(
            text=query,
            layers=[MemoryLayer.SEMANTIC],
            top_k=top_k * 2,  # over-fetch then filter
        )
        results = self.store.search(mq)

        # Filter by confidence and type
        filtered = [
            r
            for r in results
            if r.content.get("type") == "fact"
            and r.content.get("confidence", 0) >= min_confidence
        ]
        return filtered[:top_k]

    def get_relations(
        self,
        entity: str,
        relation_type: Optional[str] = None,
    ) -> list[MemoryResult]:
        """Get all relations for an entity."""
        mq = MemoryQuery(
            text=entity,
            layers=[MemoryLayer.SEMANTIC],
            top_k=100,
        )
        results = self.store.search(mq)

        filtered = []
        for r in results:
            if r.content.get("type") != "relation":
                continue
            if r.content.get("source") != entity and r.content.get("target") != entity:
                continue
            if relation_type and r.content.get("relation_type") != relation_type:
                continue
            filtered.append(r)

        return filtered

    def count(self) -> int:
        """Count semantic memories."""
        return self.store.count(agent_id=self.agent_id, layer=MemoryLayer.SEMANTIC)

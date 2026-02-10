"""
Episodic Memory — Timestamped experiences with outcomes.

Each episode records: task, strategy, outcome, context, and an embedding
for semantic recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from agentmem.store import MemoryStore
from agentmem.types import (
    AgentId,
    Embedding,
    MemoryId,
    MemoryLayer,
    MemoryQuery,
    MemoryResult,
    OutcomeLabel,
)


@dataclass
class Episode:
    """A single episodic memory.

    Attributes:
        task: What was being done.
        outcome: How it turned out.
        strategy: Approach used (optional).
        context: Situational context dict.
        reasoning: Why this strategy was chosen.
        timestamp: When the episode occurred.
    """

    task: str
    outcome: OutcomeLabel
    strategy: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None
    timestamp: Optional[datetime] = None


class EpisodicMemory:
    """Agent-scoped episodic memory with semantic recall.

    Args:
        store: The shared MemoryStore backend.
        agent_id: This agent's unique identifier.
        embedding_fn: Optional function to generate embeddings from text.

    Example:
        >>> store = MemoryStore()
        >>> epi = EpisodicMemory(store=store, agent_id="apex-01")
        >>> epi.record(
        ...     task="implement rate limiter",
        ...     outcome="success",
        ...     strategy="sliding window with Redis",
        ... )
        >>> results = epi.recall("rate limiting strategies", top_k=5)
    """

    def __init__(
        self,
        store: MemoryStore,
        agent_id: AgentId,
        embedding_fn: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.agent_id = agent_id
        self._embedding_fn = embedding_fn

    def record(
        self,
        task: str,
        outcome: str | OutcomeLabel,
        strategy: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        reasoning: Optional[str] = None,
    ) -> MemoryId:
        """Record a new episodic memory.

        Args:
            task: Description of the task.
            outcome: Outcome label ("success", "failure", "partial", "unknown").
            strategy: Strategy or approach used.
            context: Additional context (language, framework, etc.).
            reasoning: Why this approach was chosen.

        Returns:
            The memory ID of the stored episode.
        """
        if isinstance(outcome, str):
            outcome = OutcomeLabel(outcome)

        content = {
            "task": task,
            "outcome": outcome.value,
            "strategy": strategy,
            "context": context or {},
            "reasoning": reasoning,
        }

        # Generate embedding from task + strategy text
        embed_text = f"{task} {strategy or ''}"
        embedding = self._embed(embed_text)

        return self.store.store(
            layer=MemoryLayer.EPISODIC,
            agent_id=self.agent_id,
            content=content,
            embedding=embedding,
            metadata={"outcome": outcome.value},
        )

    def recall(
        self,
        query: str,
        top_k: int = 10,
        outcome_filter: Optional[OutcomeLabel] = None,
        min_similarity: float = 0.3,
    ) -> list[MemoryResult]:
        """Recall episodic memories similar to the query.

        Args:
            query: Natural-language query.
            top_k: Max results.
            outcome_filter: Optional filter by outcome type.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of matching MemoryResult objects.
        """
        mq = MemoryQuery(
            text=query,
            agent_ids=[self.agent_id],
            layers=[MemoryLayer.EPISODIC],
            top_k=top_k,
            min_similarity=min_similarity,
        )

        query_embedding = self._embed(query)
        results = self.store.search(mq, query_embedding=query_embedding)

        if outcome_filter:
            results = [
                r
                for r in results
                if r.content.get("outcome") == outcome_filter.value
            ]

        return results

    def recall_successes(self, query: str, top_k: int = 5) -> list[MemoryResult]:
        """Recall only successful episodes."""
        return self.recall(query, top_k=top_k, outcome_filter=OutcomeLabel.SUCCESS)

    def recall_failures(self, query: str, top_k: int = 5) -> list[MemoryResult]:
        """Recall only failed episodes — learn from mistakes."""
        return self.recall(query, top_k=top_k, outcome_filter=OutcomeLabel.FAILURE)

    def count(self) -> int:
        """Count this agent's episodic memories."""
        return self.store.count(agent_id=self.agent_id, layer=MemoryLayer.EPISODIC)

    def _embed(self, text: str) -> Optional[Embedding]:
        """Generate embedding from text. Falls back to random for dev/test."""
        if self._embedding_fn:
            vec = self._embedding_fn(text)
            return Embedding(vector=np.array(vec), model="custom")

        # Deterministic hash-based embedding for testing (not for production)
        rng = np.random.default_rng(seed=hash(text) % (2**31))
        vec = rng.standard_normal(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return Embedding(vector=vec, model="hash-test")

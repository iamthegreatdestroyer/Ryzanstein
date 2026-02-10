"""
ConsolidationPipeline — Cross-agent pattern extraction and generalization.

Periodically processes episodic memories from multiple agents to:
1. Extract cross-agent patterns
2. Resolve contradictions
3. Generalize episodes into semantic knowledge
4. Promote successful strategies to procedural memory
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from agentmem.semantic import Fact, SemanticMemory
from agentmem.store import MemoryStore
from agentmem.types import AgentId, MemoryLayer, MemoryQuery, OutcomeLabel


@dataclass
class ConsolidationResult:
    """Result of a consolidation run.

    Attributes:
        episodes_processed: Number of episodes examined.
        facts_extracted: Number of new facts added to semantic memory.
        contradictions_found: Number of contradictions detected.
        patterns_discovered: Number of cross-agent patterns found.
    """

    episodes_processed: int = 0
    facts_extracted: int = 0
    contradictions_found: int = 0
    patterns_discovered: int = 0


class ConsolidationPipeline:
    """Cross-agent memory consolidation.

    Args:
        store: Shared MemoryStore.
        semantic: Optional SemanticMemory for storing extracted knowledge.
        min_pattern_count: Minimum occurrences for pattern promotion.
        contradiction_threshold: Similarity above which opposing outcomes = contradiction.

    Example:
        >>> store = MemoryStore()
        >>> pipeline = ConsolidationPipeline(store=store)
        >>> result = pipeline.consolidate(
        ...     agent_ids=["apex-01", "cipher-02", "fortress-08"]
        ... )
        >>> print(f"Discovered {result.patterns_discovered} patterns")
    """

    def __init__(
        self,
        store: MemoryStore,
        semantic: Optional[SemanticMemory] = None,
        min_pattern_count: int = 2,
        contradiction_threshold: float = 0.85,
    ) -> None:
        self.store = store
        self.semantic = semantic or SemanticMemory(store=store, agent_id="__consolidated__")
        self.min_pattern_count = min_pattern_count
        self.contradiction_threshold = contradiction_threshold

    def consolidate(
        self,
        agent_ids: Optional[list[AgentId]] = None,
        max_episodes: int = 1000,
    ) -> ConsolidationResult:
        """Run consolidation across agent episodic memories.

        Args:
            agent_ids: Specific agents to consolidate (None = all).
            max_episodes: Maximum episodes to process per run.

        Returns:
            ConsolidationResult with processing statistics.
        """
        result = ConsolidationResult()

        # 1. Gather episodic memories
        query = MemoryQuery(
            text="",
            agent_ids=agent_ids,
            layers=[MemoryLayer.EPISODIC],
            top_k=max_episodes,
            min_similarity=0.0,
        )
        episodes = self.store.search(query)
        result.episodes_processed = len(episodes)

        if not episodes:
            return result

        # 2. Extract strategy→outcome patterns
        strategy_outcomes: dict[str, list[str]] = defaultdict(list)
        task_strategies: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for ep in episodes:
            strategy = ep.content.get("strategy", "")
            outcome = ep.content.get("outcome", "unknown")
            task = ep.content.get("task", "")

            if strategy:
                strategy_outcomes[strategy].append(outcome)
            if task:
                task_strategies[task].append(
                    {"strategy": strategy, "outcome": outcome, "agent": ep.agent_id}
                )

        # 3. Discover cross-agent patterns
        for strategy, outcomes in strategy_outcomes.items():
            if len(outcomes) < self.min_pattern_count:
                continue

            counts = Counter(outcomes)
            total = len(outcomes)
            success_rate = counts.get("success", 0) / total

            fact = Fact(
                subject=strategy,
                predicate="has_success_rate",
                value=f"{success_rate:.2f}",
                confidence=min(total / 10.0, 1.0),  # More data = more confident
            )
            self.semantic.add_fact(fact)
            result.facts_extracted += 1
            result.patterns_discovered += 1

        # 4. Detect contradictions (same task, opposite outcomes)
        for task, strategies in task_strategies.items():
            outcomes_set = {s["outcome"] for s in strategies}
            if "success" in outcomes_set and "failure" in outcomes_set:
                result.contradictions_found += 1
                # Record the contradiction as a fact
                self.semantic.add_fact(
                    Fact(
                        subject=task,
                        predicate="has_contradictory_outcomes",
                        value="success_and_failure_observed",
                        confidence=0.8,
                    )
                )
                result.facts_extracted += 1

        return result

"""Ryzanstein integration for agentmem.

Provides an async client that connects to the Ryzanstein LLM server
for AI-enhanced memory operations:
- Semantic consolidation (summarising episodic → semantic)
- Relevance scoring for memory retrieval
- Natural-language memory querying
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from agentmem.types import AgentId, MemoryQuery, MemoryResult


@dataclass
class RyzansteinConfig:
    """Connection settings for the Ryzanstein API server."""

    url: str = "http://localhost:8000"
    timeout_secs: float = 30.0
    model: str = "ryzanstein"


class RyzansteinMemoryClient:
    """Bridge between agentmem stores and Ryzanstein LLM capabilities."""

    def __init__(self, config: RyzansteinConfig | None = None) -> None:
        self.config = config or RyzansteinConfig()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if the Ryzanstein server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.config.url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def consolidate_episodes(
        self,
        agent_id: AgentId,
        episodes: list[dict[str, Any]],
    ) -> str:
        """Ask Ryzanstein to summarise a batch of episodes into a semantic fact."""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_secs) as client:
                resp = await client.post(
                    f"{self.config.url}/v1/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You consolidate agent memory. Given raw episodes, "
                                    "extract the key fact or insight in one sentence."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Agent: {agent_id}\n"
                                    f"Episodes ({len(episodes)}):\n"
                                    + "\n".join(
                                        str(e)[:200] for e in episodes[:10]
                                    )
                                ),
                            },
                        ],
                    },
                )
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            return self._fallback_consolidation(agent_id, episodes)

    # ------------------------------------------------------------------
    # Relevance Scoring
    # ------------------------------------------------------------------

    async def score_relevance(
        self,
        query: MemoryQuery,
        candidates: list[MemoryResult],
    ) -> list[float]:
        """Use Ryzanstein to re-rank memory candidates by relevance.

        Returns a list of scores ∈ [0, 1] parallel to *candidates*.
        Falls back to uniform scores on failure.
        """
        try:
            texts = [str(c) for c in candidates]
            async with httpx.AsyncClient(timeout=self.config.timeout_secs) as client:
                resp = await client.post(
                    f"{self.config.url}/v1/embeddings",
                    json={"model": self.config.model, "input": [str(query)] + texts},
                )
                data = resp.json()
                embeddings = [e["embedding"] for e in data["data"]]
                q_emb = embeddings[0]
                return [
                    self._cosine(q_emb, emb) for emb in embeddings[1:]
                ]
        except Exception:
            return [0.5] * len(candidates)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _fallback_consolidation(
        agent_id: AgentId, episodes: list[dict[str, Any]]
    ) -> str:
        n = len(episodes)
        return f"Agent {agent_id}: {n} episodes recorded (consolidation offline)."

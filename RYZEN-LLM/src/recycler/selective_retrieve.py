"""
Selective RSU Retrieval
[REF:TR-006e] - Token Recycling System: Query-Aware Retrieval

SCOPE NOTE (2026-07-01): implements single-best-match retrieval for the
answer-cache tier (semantic_compress + vector_bank). multi_stage_retrieve and
diversity/MMR ranking apply to multi-RSU context assembly for the token/KV
-cache tier — deferred alongside density_analyzer.py / context_injector.py
until a real forward-pass engine makes that tier meaningful. Relevance
scoring is not re-implemented here since Qdrant's search already returns a
cosine score per hit.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single retrieved RSU with its similarity score."""
    rsu_id: str
    prompt: str
    answer: str
    model: str
    score: float
    metadata: Dict[str, Any]


class SelectiveRetriever:
    """Retrieves the best-matching RSU for a query embedding."""

    def __init__(self, vector_bank: Any, top_k: int = 1):
        self.vector_bank = vector_bank
        self.top_k = top_k

    async def retrieve(
        self,
        query_embedding: List[float],
        score_threshold: float = 0.95,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[RetrievalResult]:
        hits = await self.vector_bank.retrieve(
            query_embedding,
            limit=self.top_k,
            score_threshold=score_threshold,
            filter_dict=filter_dict,
        )
        if not hits:
            return None
        top = hits[0]
        payload = top.get("payload", {}) or {}
        known = ("prompt", "answer", "model", "created_at")
        return RetrievalResult(
            rsu_id=top["id"],
            prompt=payload.get("prompt", ""),
            answer=payload.get("answer", ""),
            model=payload.get("model", ""),
            score=top.get("score", 0.0),
            metadata={k: v for k, v in payload.items() if k not in known},
        )

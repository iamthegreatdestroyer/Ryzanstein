"""
Selective RSU Retrieval
[REF:TR-006e] - Token Recycling System: Query-Aware Retrieval

Migrated 2026-07-02: consumes sigma_core Hit objects from VectorBank.retrieve
(which now delegates to sigma_core.QdrantStore.asearch) instead of raw dicts.

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

    @staticmethod
    def _to_result(hit: Any) -> RetrievalResult:
        payload = hit.payload or {}
        known = ("prompt", "answer", "model", "created_at")
        return RetrievalResult(
            rsu_id=hit.id,
            prompt=payload.get("prompt", ""),
            answer=payload.get("answer", ""),
            model=payload.get("model", ""),
            score=hit.score,
            metadata={k: v for k, v in payload.items() if k not in known},
        )

    async def retrieve_candidates(
        self,
        query_embedding: List[float],
        score_threshold: float = 0.95,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Return up to `limit` (default self.top_k) nearest RSUs above
        score_threshold, mapped to RetrievalResult, preserving Qdrant's
        descending-cosine order.

        Unlike retrieve(), does NOT discard the tail: callers that need to
        filter expired candidates or apply a margin/ambiguity gate over the
        neighbourhood (server.py's lookup) consume the full list. Fetching >1
        candidate is what lets an expired nearest be skipped in favour of a
        fresh runner-up instead of vetoing the whole lookup.
        """
        k = self.top_k if limit is None else limit
        hits = await self.vector_bank.retrieve(
            query_embedding,
            limit=k,
            score_threshold=score_threshold,
            filter_dict=filter_dict,
        )
        return [self._to_result(h) for h in (hits or [])]

    async def retrieve(
        self,
        query_embedding: List[float],
        score_threshold: float = 0.95,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[RetrievalResult]:
        """Single best match -- back-compat wrapper over retrieve_candidates()."""
        cands = await self.retrieve_candidates(
            query_embedding, score_threshold=score_threshold,
            filter_dict=filter_dict, limit=self.top_k,
        )
        return cands[0] if cands else None

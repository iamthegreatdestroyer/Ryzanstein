"""
Vector Bank - RSU Storage and Retrieval
[REF:TR-006c] - Token Recycling System: Vector Database

Persists RSUs in Qdrant for similarity-based retrieval.

Migrated 2026-07-02 onto sigma-core: this class was a hand-rolled Qdrant REST
client duplicating sigma_core.retrieval.QdrantStore. It now delegates to the
shared fabric client (async aadd/asearch/adelete/acount), keeping its
RSU-shaped public API so SemanticCompressor/SelectiveRetriever/server.py are
unchanged. retrieve() now returns sigma_core Hit objects (SelectiveRetriever
was updated to match). RSU ids are uuid4 → pass through QdrantStore._to_qid
unchanged, so the existing live rsu_bank collection stays compatible.
"""

from typing import Any, Dict, List, Optional

from sigma_core.retrieval import QdrantStore, Hit


class VectorBank:
    """Manages storage and retrieval of RSUs in a Qdrant collection (via sigma-core)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rsu_bank",
        vector_size: int = 768,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._store = QdrantStore(url=f"http://{host}:{port}", dim=vector_size)

    async def store(self, rsu: Any) -> str:
        """Store an RSU in Qdrant. Returns the RSU's id."""
        payload = {
            "prompt": rsu.prompt,
            "answer": rsu.answer,
            "model": rsu.model,
            "created_at": rsu.created_at,
            **rsu.metadata,
        }
        return await self._store.aadd(
            self.collection_name, id=rsu.id, vector=rsu.embedding, payload=payload
        )

    async def retrieve(
        self,
        query_embedding: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Hit]:
        """Return the top-`limit` nearest RSUs as sigma_core Hit objects."""
        return await self._store.asearch(
            self.collection_name,
            query_embedding,
            k=limit,
            score_threshold=score_threshold,
            filter=filter_dict,
        )

    async def delete(self, rsu_id: str) -> bool:
        return await self._store.adelete(self.collection_name, rsu_id)

    async def count(self) -> int:
        return await self._store.acount(self.collection_name)

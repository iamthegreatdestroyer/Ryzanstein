"""
Vector Bank - RSU Storage and Retrieval
[REF:TR-006c] - Token Recycling System: Vector Database

Persists RSUs in Qdrant for similarity-based retrieval. Implemented via plain
REST calls (httpx) rather than the qdrant-client SDK — the box's system
Python is externally-managed (PEP 668 / Debian 13), so this avoids adding a
new pip dependency entirely; httpx is already a dependency of this service.
"""

from typing import Any, Dict, List, Optional
import httpx


class VectorBank:
    """Manages storage and retrieval of RSUs in a Qdrant collection."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rsu_bank",
        vector_size: int = 768,
    ):
        self.base_url = f"http://{host}:{port}"
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensured = False

    async def _ensure_collection(self, client: httpx.AsyncClient) -> None:
        if self._ensured:
            return
        resp = await client.get(f"{self.base_url}/collections/{self.collection_name}")
        if resp.status_code != 200:
            await client.put(
                f"{self.base_url}/collections/{self.collection_name}",
                json={"vectors": {"size": self.vector_size, "distance": "Cosine"}},
            )
        self._ensured = True

    async def store(self, rsu: Any) -> str:
        """Store an RSU in Qdrant. Returns the RSU's id."""
        payload = {
            "prompt": rsu.prompt,
            "answer": rsu.answer,
            "model": rsu.model,
            "created_at": rsu.created_at,
            **rsu.metadata,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._ensure_collection(client)
            resp = await client.put(
                f"{self.base_url}/collections/{self.collection_name}/points",
                json={"points": [{"id": rsu.id, "vector": rsu.embedding, "payload": payload}]},
            )
            resp.raise_for_status()
        return rsu.id

    async def retrieve(
        self,
        query_embedding: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-`limit` nearest points (Qdrant's raw result shape)."""
        body: Dict[str, Any] = {"vector": query_embedding, "limit": limit, "with_payload": True}
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        if filter_dict:
            body["filter"] = filter_dict
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._ensure_collection(client)
            resp = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=body,
            )
            resp.raise_for_status()
            return resp.json().get("result", [])

    async def delete(self, rsu_id: str) -> bool:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/delete",
                json={"points": [rsu_id]},
            )
            return resp.status_code == 200

    async def count(self) -> int:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/collections/{self.collection_name}")
            if resp.status_code != 200:
                return 0
            return resp.json().get("result", {}).get("points_count", 0) or 0

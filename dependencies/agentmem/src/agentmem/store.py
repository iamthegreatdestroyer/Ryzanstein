"""
MemoryStore — Unified storage backend for all memory layers.

Supports pluggable backends: local (in-process), Redis, Neo4j, ΣVAULT.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from agentmem.types import (
    AgentId,
    Embedding,
    MemoryId,
    MemoryLayer,
    MemoryQuery,
    MemoryResult,
)


@dataclass
class StoredMemory:
    """Internal representation of a stored memory."""

    memory_id: MemoryId
    layer: MemoryLayer
    agent_id: AgentId
    content: dict[str, Any]
    embedding: Optional[Embedding]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Unified memory storage with vector search support.

    Args:
        backend: Storage backend type ("local", "redis", "neo4j", "vault").
        embedding_dim: Dimension of embedding vectors.
        config: Backend-specific configuration.

    Example:
        >>> store = MemoryStore(backend="local")
        >>> mid = store.store(
        ...     layer=MemoryLayer.EPISODIC,
        ...     agent_id="apex-01",
        ...     content={"task": "rate limiting"},
        ...     embedding=some_embedding,
        ... )
        >>> results = store.search(MemoryQuery(text="rate limit"))
    """

    def __init__(
        self,
        backend: str = "local",
        embedding_dim: int = 384,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.embedding_dim = embedding_dim
        self.config = config or {}

        # Local in-memory storage
        self._memories: dict[MemoryId, StoredMemory] = {}
        self._by_agent: dict[AgentId, list[MemoryId]] = defaultdict(list)
        self._by_layer: dict[MemoryLayer, list[MemoryId]] = defaultdict(list)

    def store(
        self,
        layer: MemoryLayer,
        agent_id: AgentId,
        content: dict[str, Any],
        embedding: Optional[Embedding] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryId:
        """Store a memory and return its ID.

        Args:
            layer: Which memory layer to store in.
            agent_id: Owning agent identifier.
            content: Memory content as a dict.
            embedding: Optional vector embedding for similarity search.
            metadata: Optional additional metadata.

        Returns:
            The unique memory ID.
        """
        memory_id = str(uuid.uuid4())
        stored = StoredMemory(
            memory_id=memory_id,
            layer=layer,
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        self._memories[memory_id] = stored
        self._by_agent[agent_id].append(memory_id)
        self._by_layer[layer].append(memory_id)
        return memory_id

    def retrieve(self, memory_id: MemoryId) -> Optional[StoredMemory]:
        """Retrieve a specific memory by ID."""
        return self._memories.get(memory_id)

    def search(
        self,
        query: MemoryQuery,
        query_embedding: Optional[Embedding] = None,
    ) -> list[MemoryResult]:
        """Search memories using semantic similarity and filters.

        Args:
            query: The memory query specification.
            query_embedding: Optional embedding of the query text.

        Returns:
            List of MemoryResult sorted by similarity (descending).
        """
        candidates: list[StoredMemory] = []

        for mem in self._memories.values():
            # Filter by agent
            if query.agent_ids and mem.agent_id not in query.agent_ids:
                continue
            # Filter by layer
            if query.layers and mem.layer not in query.layers:
                continue
            # Filter by time range
            if query.time_range:
                start, end = query.time_range
                if mem.timestamp < start or mem.timestamp > end:
                    continue
            candidates.append(mem)

        # Score by embedding similarity if available
        results: list[MemoryResult] = []
        for mem in candidates:
            similarity = 0.0
            if query_embedding and mem.embedding:
                similarity = query_embedding.cosine_similarity(mem.embedding)
                if similarity < query.min_similarity:
                    continue
            elif query.text:
                # Fallback: text containment check
                content_str = str(mem.content).lower()
                if query.text.lower() in content_str:
                    similarity = 0.75
                else:
                    continue

            results.append(
                MemoryResult(
                    memory_id=mem.memory_id,
                    layer=mem.layer,
                    agent_id=mem.agent_id,
                    content=mem.content,
                    similarity=similarity,
                    timestamp=mem.timestamp,
                    metadata=mem.metadata,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[: query.top_k]

    def delete(self, memory_id: MemoryId) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        mem = self._memories.pop(memory_id, None)
        if mem is None:
            return False
        self._by_agent[mem.agent_id] = [
            m for m in self._by_agent[mem.agent_id] if m != memory_id
        ]
        self._by_layer[mem.layer] = [
            m for m in self._by_layer[mem.layer] if m != memory_id
        ]
        return True

    def count(
        self,
        agent_id: Optional[AgentId] = None,
        layer: Optional[MemoryLayer] = None,
    ) -> int:
        """Count memories, optionally filtered by agent and/or layer."""
        if agent_id and layer:
            return sum(
                1
                for mid in self._by_agent.get(agent_id, [])
                if self._memories[mid].layer == layer
            )
        if agent_id:
            return len(self._by_agent.get(agent_id, []))
        if layer:
            return len(self._by_layer.get(layer, []))
        return len(self._memories)

    def clear(self, agent_id: Optional[AgentId] = None) -> int:
        """Clear all memories (or all for a specific agent). Returns count deleted."""
        if agent_id:
            mids = self._by_agent.pop(agent_id, [])
            for mid in mids:
                mem = self._memories.pop(mid, None)
                if mem:
                    self._by_layer[mem.layer] = [
                        m for m in self._by_layer[mem.layer] if m != mid
                    ]
            return len(mids)

        count = len(self._memories)
        self._memories.clear()
        self._by_agent.clear()
        self._by_layer.clear()
        return count

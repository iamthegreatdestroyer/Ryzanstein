"""Core type definitions for agentmem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np


# --- Identifiers ---

AgentId = str
MemoryId = str


class MemoryLayer(Enum):
    """Memory layers in the four-layer architecture."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class OutcomeLabel(Enum):
    """Outcome labels for episodic memories."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


# --- Query types ---


@dataclass
class MemoryQuery:
    """Query across memory layers.

    Attributes:
        text: Natural-language query string.
        agent_ids: Filter to specific agents (None = all).
        layers: Which layers to search (None = all).
        top_k: Maximum results to return.
        min_similarity: Minimum cosine similarity threshold.
        time_range: Optional (start, end) datetime filter.
    """

    text: str
    agent_ids: Optional[list[AgentId]] = None
    layers: Optional[list[MemoryLayer]] = None
    top_k: int = 10
    min_similarity: float = 0.5
    time_range: Optional[tuple[datetime, datetime]] = None


@dataclass
class MemoryResult:
    """A single result from memory recall.

    Attributes:
        memory_id: Unique identifier.
        layer: Which memory layer this came from.
        agent_id: Which agent owns this memory.
        content: The memory content (varies by layer).
        similarity: Cosine similarity to query.
        timestamp: When the memory was created.
        metadata: Additional metadata.
    """

    memory_id: MemoryId
    layer: MemoryLayer
    agent_id: AgentId
    content: dict[str, Any]
    similarity: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Embedding type ---


@dataclass
class Embedding:
    """A dense vector embedding with provenance.

    Attributes:
        vector: The embedding vector (numpy array).
        model: Which model generated this embedding.
        dimension: Embedding dimensionality.
    """

    vector: np.ndarray
    model: str = "default"
    dimension: int = 384

    def cosine_similarity(self, other: Embedding) -> float:
        """Compute cosine similarity with another embedding."""
        dot = float(np.dot(self.vector, other.vector))
        norm_a = float(np.linalg.norm(self.vector))
        norm_b = float(np.linalg.norm(other.vector))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

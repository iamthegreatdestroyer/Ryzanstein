"""
Semantic Compression Module
[REF:TR-006b] - Token Recycling System: RSU Compression

Compresses a (prompt, answer) pair into a Recyclable Semantic Unit (RSU) keyed
by embedding similarity, so a semantically near-identical future prompt can be
served from cache instead of re-running inference.

SCOPE NOTE (2026-07-01): the original TR-006 design compressed raw token
sequences for in-engine KV-cache reuse (see basic_recycler.cpp,
density_analyzer.py, context_injector.py). That tier requires real model
attention weights, unavailable while Ryzanstein proxies Ollama rather than
running its own forward pass. This module is activated at the ANSWER-CACHE
granularity instead (embed the prompt, cache the full response) — the same
pattern already proven in In My Head's semantic_cache.py. Revisit token/KV
-cache-level recycling once a real forward-pass engine exists.
"""

from typing import Awaitable, Callable, Dict, Any, List, Optional
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RSU:
    """Recyclable Semantic Unit — a cached (prompt, answer) pair keyed by embedding."""
    id: str
    embedding: List[float]
    prompt: str
    answer: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


EmbedFn = Callable[[str], Awaitable[List[float]]]


class SemanticCompressor:
    """
    Compresses a (prompt, answer) pair into an RSU using an injected async
    embedding function — kept decoupled from any specific backend so the
    Ryzanstein gateway can wire this to its own Ollama-backed embed call.
    """

    def __init__(self, embed_fn: EmbedFn):
        self.embed_fn = embed_fn

    async def compress(
        self,
        prompt: str,
        answer: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RSU:
        embedding = await self.embed_fn(prompt)
        return RSU(
            id=str(uuid.uuid4()),
            embedding=embedding,
            prompt=prompt,
            answer=answer,
            model=model,
            metadata=metadata or {},
        )

    async def embed_query(self, text: str) -> List[float]:
        """Embed a query for retrieval (no RSU wrapping needed)."""
        return await self.embed_fn(text)

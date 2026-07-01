"""
Token Recycling System Package
[REF:TR-006] - Token Recycling System

Answer-level semantic caching, activated 2026-07-01 against Qdrant (see
semantic_compress.py, vector_bank.py, selective_retrieve.py). density_analyzer
and context_injector remain unimplemented — they require real model attention
weights, unavailable while Ryzanstein proxies Ollama rather than running its
own forward pass. See each module's docstring for when to revisit.

Modules:
    density_analyzer: Token density scoring and selection (DEFERRED)
    semantic_compress: RSU compression and embedding (ACTIVE)
    vector_bank: RSU storage and retrieval with Qdrant (ACTIVE)
    context_injector: Context reconstruction from RSUs (DEFERRED)
    selective_retrieve: Query-aware RSU retrieval (ACTIVE)
"""

__version__ = "0.2.0"
__author__ = "Ryzanstein LLM Project"

from .semantic_compress import SemanticCompressor, RSU
from .vector_bank import VectorBank
from .selective_retrieve import SelectiveRetriever, RetrievalResult

__all__ = [
    "SemanticCompressor",
    "RSU",
    "VectorBank",
    "SelectiveRetriever",
    "RetrievalResult",
]

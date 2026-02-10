"""
agentmem — Cross-Agent Episodic Memory Protocol

Four-layer memory architecture for multi-agent systems:
- Working Memory (ephemeral task context)
- Episodic Memory (timestamped experiences)
- Semantic Memory (knowledge graph)
- Procedural Memory (learned workflows)
"""

from agentmem.store import MemoryStore
from agentmem.episodic import EpisodicMemory, Episode
from agentmem.semantic import SemanticMemory, Fact, Relation
from agentmem.procedural import ProceduralMemory, Workflow
from agentmem.working import WorkingMemory
from agentmem.consolidation import ConsolidationPipeline
from agentmem.types import AgentId, MemoryQuery, MemoryResult
from agentmem.ryzanstein import RyzansteinMemoryClient, RyzansteinConfig

__all__ = [
    "MemoryStore",
    "EpisodicMemory",
    "Episode",
    "SemanticMemory",
    "Fact",
    "Relation",
    "ProceduralMemory",
    "Workflow",
    "WorkingMemory",
    "ConsolidationPipeline",
    "AgentId",
    "MemoryQuery",
    "MemoryResult",
    "RyzansteinMemoryClient",
    "RyzansteinConfig",
]

__version__ = "0.1.0"

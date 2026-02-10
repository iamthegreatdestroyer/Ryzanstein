"""
Procedural Memory — Learned workflows and strategy templates.

Stores reusable multi-step procedures discovered through experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentmem.store import MemoryStore
from agentmem.types import AgentId, MemoryId, MemoryLayer, MemoryQuery, MemoryResult


@dataclass
class WorkflowStep:
    """A single step in a procedural workflow.

    Attributes:
        action: What to do.
        tool: Which tool/agent to use.
        parameters: Parameters for the action.
        expected_outcome: What success looks like.
    """

    action: str
    tool: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None


@dataclass
class Workflow:
    """A learned multi-step procedure.

    Attributes:
        name: Short name for the workflow.
        description: What this workflow accomplishes.
        steps: Ordered list of steps.
        applicability: When this workflow should be applied.
        success_rate: Historical success rate.
        usage_count: How many times this has been used.
    """

    name: str
    description: str
    steps: list[WorkflowStep]
    applicability: str = ""
    success_rate: float = 0.0
    usage_count: int = 0


class ProceduralMemory:
    """Storage and retrieval for learned workflows.

    Args:
        store: Shared MemoryStore backend.
        agent_id: Agent identifier.

    Example:
        >>> store = MemoryStore()
        >>> proc = ProceduralMemory(store=store, agent_id="flux-11")
        >>> proc.add_workflow(Workflow(
        ...     name="deploy-k8s",
        ...     description="Deploy to Kubernetes cluster",
        ...     steps=[WorkflowStep(action="build image"), ...],
        ... ))
    """

    def __init__(self, store: MemoryStore, agent_id: AgentId) -> None:
        self.store = store
        self.agent_id = agent_id

    def add_workflow(self, workflow: Workflow) -> MemoryId:
        """Store a learned workflow."""
        content = {
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "action": s.action,
                    "tool": s.tool,
                    "parameters": s.parameters,
                    "expected_outcome": s.expected_outcome,
                }
                for s in workflow.steps
            ],
            "applicability": workflow.applicability,
            "success_rate": workflow.success_rate,
            "usage_count": workflow.usage_count,
        }
        return self.store.store(
            layer=MemoryLayer.PROCEDURAL,
            agent_id=self.agent_id,
            content=content,
            metadata={"success_rate": workflow.success_rate},
        )

    def find_workflow(
        self,
        query: str,
        top_k: int = 5,
        min_success_rate: float = 0.0,
    ) -> list[MemoryResult]:
        """Find applicable workflows for a task."""
        mq = MemoryQuery(
            text=query,
            agent_ids=[self.agent_id],
            layers=[MemoryLayer.PROCEDURAL],
            top_k=top_k * 2,
        )
        results = self.store.search(mq)

        filtered = [
            r
            for r in results
            if r.content.get("success_rate", 0) >= min_success_rate
        ]
        return filtered[:top_k]

    def update_success_rate(
        self, memory_id: MemoryId, succeeded: bool
    ) -> Optional[float]:
        """Update a workflow's success rate. Returns new rate or None."""
        mem = self.store.retrieve(memory_id)
        if mem is None:
            return None

        count = mem.content.get("usage_count", 0) + 1
        old_rate = mem.content.get("success_rate", 0.0)
        # Incremental mean update
        new_rate = old_rate + (1.0 if succeeded else 0.0 - old_rate) / count

        mem.content["usage_count"] = count
        mem.content["success_rate"] = new_rate
        return new_rate

    def count(self) -> int:
        """Count procedural memories."""
        return self.store.count(agent_id=self.agent_id, layer=MemoryLayer.PROCEDURAL)

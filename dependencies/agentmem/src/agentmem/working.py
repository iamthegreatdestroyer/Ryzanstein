"""
Working Memory — Ephemeral current-task context.

Short-lived memory that holds the active task context for an agent.
Automatically cleared when a new task starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentmem.types import AgentId


@dataclass
class WorkingContext:
    """Current working memory state.

    Attributes:
        task: Active task description.
        goal: What the agent is trying to achieve.
        variables: Key-value pairs for the current task.
        scratchpad: Intermediate reasoning/notes.
        focus_stack: Stack of sub-tasks being worked on.
    """

    task: Optional[str] = None
    goal: Optional[str] = None
    variables: dict[str, Any] = field(default_factory=dict)
    scratchpad: list[str] = field(default_factory=list)
    focus_stack: list[str] = field(default_factory=list)


class WorkingMemory:
    """Ephemeral working memory for the current task.

    Args:
        agent_id: This agent's identifier.
        capacity: Maximum scratchpad entries before eviction.

    Example:
        >>> wm = WorkingMemory(agent_id="apex-01")
        >>> wm.begin_task("implement caching", goal="Add Redis cache layer")
        >>> wm.set("cache_backend", "redis")
        >>> wm.note("User prefers async approach")
    """

    def __init__(self, agent_id: AgentId, capacity: int = 50) -> None:
        self.agent_id = agent_id
        self.capacity = capacity
        self._context = WorkingContext()

    def begin_task(self, task: str, goal: Optional[str] = None) -> None:
        """Clear working memory and start a new task."""
        self._context = WorkingContext(task=task, goal=goal)

    @property
    def task(self) -> Optional[str]:
        """Current active task."""
        return self._context.task

    @property
    def goal(self) -> Optional[str]:
        """Current goal."""
        return self._context.goal

    def set(self, key: str, value: Any) -> None:
        """Store a variable in working memory."""
        self._context.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a variable from working memory."""
        return self._context.variables.get(key, default)

    def note(self, text: str) -> None:
        """Add a scratchpad note. Evicts oldest if at capacity."""
        if len(self._context.scratchpad) >= self.capacity:
            self._context.scratchpad.pop(0)
        self._context.scratchpad.append(text)

    def push_focus(self, subtask: str) -> None:
        """Push a sub-task onto the focus stack."""
        self._context.focus_stack.append(subtask)

    def pop_focus(self) -> Optional[str]:
        """Pop the current sub-task. Returns None if stack is empty."""
        if not self._context.focus_stack:
            return None
        return self._context.focus_stack.pop()

    @property
    def current_focus(self) -> Optional[str]:
        """The current sub-task (top of stack)."""
        return self._context.focus_stack[-1] if self._context.focus_stack else None

    def snapshot(self) -> dict[str, Any]:
        """Export working memory as a dict for context injection."""
        return {
            "agent_id": self.agent_id,
            "task": self._context.task,
            "goal": self._context.goal,
            "variables": dict(self._context.variables),
            "scratchpad": list(self._context.scratchpad),
            "focus_stack": list(self._context.focus_stack),
        }

    def clear(self) -> None:
        """Clear all working memory."""
        self._context = WorkingContext()

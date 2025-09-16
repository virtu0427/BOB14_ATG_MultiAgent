"""Core abstractions for the minimal Google ADK runtime used in this project."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional


class Role(str, Enum):
    """Conversation roles used in the orchestrator runtime."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ORCHESTRATOR = "orchestrator"


@dataclass
class Message:
    """Represents a single message exchanged between agents."""

    role: Role
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self, **overrides: Any) -> "Message":
        data = {"role": self.role, "content": self.content, "metadata": dict(self.metadata)}
        data.update(overrides)
        if "metadata" in data:
            data["metadata"] = dict(data["metadata"])
        return Message(**data)


class Agent:
    """Base class for any agent that participates in the runtime."""

    def __init__(self, name: str, description: str, instructions: Optional[str] = None) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions or ""

    def run(self, task: str, context: Iterable[Message]) -> Message:
        """Execute the agent on the provided task."""

        raise NotImplementedError("Agents must implement the run() method")

    # Convenience API -----------------------------------------------------------------
    def system_message(self) -> Message:
        return Message(role=Role.SYSTEM, content=self.instructions, metadata={"agent": self.name})

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Agent(name={self.name!r})"

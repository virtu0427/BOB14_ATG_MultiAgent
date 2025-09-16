"""Simplified runtime loop for orchestrating agents."""
from __future__ import annotations

from typing import Iterable, List

from .core import Agent, Message, Role


class AgentRuntime:
    """Maintains conversation state and delegates work to the orchestrator agent."""

    def __init__(self, orchestrator: Agent) -> None:
        self.orchestrator = orchestrator
        self.history: List[Message] = []

    def send_user_message(self, content: str) -> Message:
        message = Message(role=Role.USER, content=content)
        self.history.append(message)
        response = self.orchestrator.run(content, list(self.history))
        self.history.append(response)
        return response

    def replay(self) -> Iterable[Message]:
        return list(self.history)

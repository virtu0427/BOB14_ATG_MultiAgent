"""Tooling primitives inspired by the Google ADK interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable


@dataclass
class Tool:
    """Represents an executable function exposed to an agent."""

    name: str
    description: str
    handler: Callable[[str], str]

    def __call__(self, input_text: str) -> str:
        return self.handler(input_text)


class ToolRegistry:
    """Simple registry that mimics the ADK tool management utilities."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Unknown tool: {name}") from exc

    def list(self) -> Iterable[Tool]:
        return self._tools.values()

    def describe(self) -> str:
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

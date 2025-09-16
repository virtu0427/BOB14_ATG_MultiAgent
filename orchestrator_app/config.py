"""Configuration helpers for the orchestrator demo."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class OllamaSettings:
    model: str = _env("OLLAMA_MODEL", "gemma3:12b")
    endpoint: str = _env("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    temperature: float = float(_env("OLLAMA_TEMPERATURE", "0.2"))


def load_ollama_settings() -> OllamaSettings:
    return OllamaSettings()

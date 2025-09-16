"""LLM client utilities tailored for the Ollama Gemma3 model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - exercised indirectly in integration usage
    import requests
except ModuleNotFoundError:  # pragma: no cover - allows running without requests installed
    requests = None  # type: ignore[assignment]


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an error."""


@dataclass
class GemmaOllamaClient:
    """Thin HTTP client for talking to a local Ollama deployment."""

    model: str = "gemma3:12b"
    endpoint: str = "http://localhost:11434/api/generate"
    timeout: int = 120

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if json_mode:
            payload["format"] = "json"
        if requests is None:  # pragma: no cover - triggered only when dependency missing
            raise OllamaError(
                "The 'requests' package is required to call the Ollama HTTP API. "
                "Install dependencies with `pip install -r requirements.txt`."
            )
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            raise OllamaError("Failed to reach the Ollama runtime") from exc
        data = response.json()
        text = data.get("response")
        if text is None:
            raise OllamaError(f"Unexpected Ollama response: {json.dumps(data)[:200]}")
        return text

    def complete_json(self, prompt: str, *, temperature: float = 0.0) -> Dict[str, Any]:
        text = self.complete(prompt, temperature=temperature, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise OllamaError("Gemma did not return valid JSON") from exc

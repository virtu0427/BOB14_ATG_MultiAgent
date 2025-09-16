from __future__ import annotations

import json

from orchestrator_app.orchestrator import build_default_orchestrator


class FakeLLM:
    def __init__(self) -> None:
        self.classification_responses: dict[str, dict[str, object]] = {}

    def register_classification(self, needle: str, agent: str, confidence: float, reasoning: str) -> None:
        self.classification_responses[needle] = {
            "agent": agent,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens=None, json_mode: bool = False):
        if json_mode:
            for needle, payload in self.classification_responses.items():
                if needle in prompt:
                    return json.dumps(payload)
            return "{}"
        if "You are Implementation Engineer" in prompt:
            return "Implementation details"
        if "You are Quality Strategist" in prompt:
            return "Test plan"
        if "You are Research Analyst" in prompt:
            return "Research summary"
        return "General response"


def test_orchestrator_routes_using_llm_json():
    llm = FakeLLM()
    llm.register_classification(
        needle="Implement a parser",
        agent="Implementation Engineer",
        confidence=0.82,
        reasoning="Coding focused",
    )
    orchestrator = build_default_orchestrator(llm=llm)
    response = orchestrator.run("Implement a parser", [])
    assert response.metadata["delegated_to"] == "Implementation Engineer"
    assert "confidence 0.82" in response.content
    assert "Implementation details" in response.content


def test_orchestrator_falls_back_to_keyword_matching():
    llm = FakeLLM()
    orchestrator = build_default_orchestrator(llm=llm)
    response = orchestrator.run("Design regression tests", [])
    assert response.metadata["delegated_to"] == "Quality Strategist"
    assert "Test plan" in response.content

"""Gemma-powered orchestrator built on top of the lightweight ADK primitives."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from google.adk import Agent, GemmaOllamaClient, Message, Role

from .agents import LLMAgent, build_specialist_agents


@dataclass
class SelectionResult:
    agent: str
    confidence: float
    reasoning: str


class GemmaTaskOrchestrator(Agent):
    """Routes tasks to specialist agents using a Gemma3 reasoning call."""

    def __init__(
        self,
        llm: GemmaOllamaClient,
        agents: Sequence[LLMAgent],
        *,
        classification_temperature: float = 0.0,
    ) -> None:
        super().__init__(
            name="Gemma Orchestrator",
            description="Delegates work to the most suitable specialist agent",
            instructions=(
                "Select the best agent for the task, explain the routing decision, "
                "and ensure the final response aggregates the agent output."
            ),
        )
        self.llm = llm
        self.agents: Dict[str, LLMAgent] = {agent.name: agent for agent in agents}
        self.classification_temperature = classification_temperature

    def _agent_directory(self) -> str:
        rows = []
        for agent in self.agents.values():
            rows.append(f"{agent.name}: {agent.description}\nInstructions: {agent.instructions}")
        return "\n\n".join(rows)

    def _build_selection_prompt(self, task: str, context: Iterable[Message]) -> str:
        transcript: List[str] = []
        for message in context:
            speaker = message.metadata.get("agent", message.role.value)
            transcript.append(f"[{speaker}] {message.content}")
        history_block = "\n".join(transcript) if transcript else "(no previous conversation)"
        prompt = (
            "You are the orchestrator deciding which specialist agent should handle a task.\n"
            "Consider the capabilities listed below and choose the best fit.\n"
            f"Available agents:\n{self._agent_directory()}\n\n"
            f"Conversation so far:\n{history_block}\n\n"
            f"Incoming task: {task}\n\n"
            "Respond with a JSON object with keys 'agent', 'confidence', and 'reasoning'.\n"
            "The confidence should be a value between 0 and 1."
        )
        return prompt

    def _parse_selection(self, response: str, task: str) -> SelectionResult:
        try:
            data = json.loads(response)
            agent_name = data.get("agent")
            if agent_name not in self.agents:
                raise KeyError
            confidence = float(data.get("confidence", 0.0))
            reasoning = str(data.get("reasoning", ""))
            return SelectionResult(agent=agent_name, confidence=confidence, reasoning=reasoning)
        except Exception:
            # Fallback heuristic based on keyword matching
            lower = task.lower()
            if any(keyword in lower for keyword in ["test", "qa", "validate", "quality"]):
                agent_name = "Quality Strategist"
            elif any(keyword in lower for keyword in ["implement", "code", "refactor", "bug"]):
                agent_name = "Implementation Engineer"
            elif any(keyword in lower for keyword in ["research", "investigate", "analyz", "plan"]):
                agent_name = "Research Analyst"
            else:
                agent_name = "Generalist Collaborator"
            return SelectionResult(agent=agent_name, confidence=0.1, reasoning="Fallback heuristic")

    def select_agent(self, task: str, context: Iterable[Message]) -> SelectionResult:
        prompt = self._build_selection_prompt(task, context)
        try:
            selection_json = self.llm.complete(
                prompt,
                temperature=self.classification_temperature,
                json_mode=True,
            )
        except Exception:
            # When the Gemma client cannot be reached, fall back to heuristics.
            selection_json = ""
        return self._parse_selection(selection_json, task)

    def run(self, task: str, context: Iterable[Message]) -> Message:
        selection = self.select_agent(task, context)
        agent = self.agents[selection.agent]
        agent_response = agent.run(task, context)
        combined_content = (
            f"Routing decision: {selection.agent} (confidence {selection.confidence:.2f}).\n"
            f"Reasoning: {selection.reasoning or 'N/A'}.\n\n"
            f"{selection.agent}'s response:\n{agent_response.content}"
        )
        return Message(
            role=Role.ORCHESTRATOR,
            content=combined_content,
            metadata={
                "agent": self.name,
                "delegated_to": selection.agent,
                "confidence": selection.confidence,
            },
        )


def build_default_orchestrator(
    *,
    llm: GemmaOllamaClient | None = None,
    classification_temperature: float = 0.0,
) -> GemmaTaskOrchestrator:
    llm = llm or GemmaOllamaClient()
    agents = build_specialist_agents(llm)
    return GemmaTaskOrchestrator(
        llm=llm,
        agents=agents,
        classification_temperature=classification_temperature,
    )

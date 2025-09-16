"""Specialist agents that the orchestrator can route work to."""
from __future__ import annotations

from typing import Iterable, List

from google.adk import Agent, GemmaOllamaClient, Message, Role


class LLMAgent(Agent):
    """Base class for simple LLM powered agents."""

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        llm: GemmaOllamaClient,
        *,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(name=name, description=description, instructions=instructions)
        self.llm = llm
        self.temperature = temperature

    def build_prompt(self, task: str, context: Iterable[Message]) -> str:
        transcript: List[str] = []
        for message in context:
            speaker = message.metadata.get("agent", message.role.value)
            transcript.append(f"[{speaker}] {message.content}")
        history_block = "\n".join(transcript) if transcript else "(no previous conversation)"
        prompt = (
            f"You are {self.name}.\n"
            f"Role description: {self.description}.\n"
            f"Operating instructions: {self.instructions}.\n"
            f"Conversation so far:\n{history_block}\n"
            f"Task: {task}\n"
            "Provide a concise, actionable response."
        )
        return prompt

    def run(self, task: str, context: Iterable[Message]) -> Message:
        prompt = self.build_prompt(task, context)
        content = self.llm.complete(prompt, temperature=self.temperature)
        return Message(role=Role.ASSISTANT, content=content, metadata={"agent": self.name})


class ResearchAgent(LLMAgent):
    def __init__(self, llm: GemmaOllamaClient) -> None:
        super().__init__(
            name="Research Analyst",
            description="Collects background knowledge and synthesises findings",
            instructions=(
                "Focus on decomposing the problem, highlight relevant background "
                "information, and cite potential research directions."
            ),
            llm=llm,
            temperature=0.1,
        )


class CodingAgent(LLMAgent):
    def __init__(self, llm: GemmaOllamaClient) -> None:
        super().__init__(
            name="Implementation Engineer",
            description="Designs and implements code changes",
            instructions=(
                "Translate requirements into code sketches, include architecture "
                "decisions, and flag dependencies or risks."
            ),
            llm=llm,
            temperature=0.15,
        )


class TestingAgent(LLMAgent):
    def __init__(self, llm: GemmaOllamaClient) -> None:
        super().__init__(
            name="Quality Strategist",
            description="Plans validation strategies and edge case coverage",
            instructions=(
                "Propose automated and manual validation steps, prioritise reliability, "
                "and enumerate measurable success criteria."
            ),
            llm=llm,
            temperature=0.15,
        )


class GeneralistAgent(LLMAgent):
    def __init__(self, llm: GemmaOllamaClient) -> None:
        super().__init__(
            name="Generalist Collaborator",
            description="Handles broad coordination or ambiguous tasks",
            instructions=(
                "Provide balanced reasoning, combine insights from research, coding, "
                "and testing perspectives, and maintain clarity."
            ),
            llm=llm,
            temperature=0.2,
        )


def build_specialist_agents(llm: GemmaOllamaClient) -> List[LLMAgent]:
    return [
        ResearchAgent(llm),
        CodingAgent(llm),
        TestingAgent(llm),
        GeneralistAgent(llm),
    ]

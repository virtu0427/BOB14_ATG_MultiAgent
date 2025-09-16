"""Command line Development UI for exercising the orchestrator."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from google.adk import AgentRuntime, GemmaOllamaClient

from .config import load_ollama_settings
from .orchestrator import build_default_orchestrator

console = Console()


def launch_chat() -> None:
    settings = load_ollama_settings()
    llm = GemmaOllamaClient(
        model=settings.model,
        endpoint=settings.endpoint,
        timeout=120,
    )
    orchestrator = build_default_orchestrator(
        llm=llm,
        classification_temperature=settings.temperature,
    )
    runtime = AgentRuntime(orchestrator)
    console.print(
        Panel(
            "Enter your task to see how the orchestrator delegates work. "
            "Type 'exit' to quit.",
            title="Gemma Orchestrator Development UI",
        )
    )
    while True:
        user_input = Prompt.ask("[bold cyan]You[/]")
        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("Exiting Development UI.")
            break
        try:
            response = runtime.send_user_message(user_input)
        except Exception as exc:  # pragma: no cover - runtime guard
            console.print(f"[bold red]Error:[/] {exc}")
            continue
        title = f"Delegated to: {response.metadata.get('delegated_to', 'unknown')}"
        console.print(Panel(response.content, title=title))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    launch_chat()

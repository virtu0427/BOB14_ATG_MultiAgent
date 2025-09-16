# BOB14_ATG_MultiAgent

This repository contains a lightweight reproduction of the Google Agent Development Kit (ADK) surface that is sufficient to build a Gemma3 orchestrator running against a local Ollama deployment. The orchestrator dynamically routes tasks to specialist agents (research, implementation, quality assurance, and generalist coordination) and exposes a command-line Development UI for manual experimentation.

## Project layout

- `google/adk/`: minimal ADK-compatible primitives (`Agent`, `Message`, `AgentRuntime`, tooling helpers, and an Ollama LLM client).
- `orchestrator_app/`: orchestrator implementation, specialist agent definitions, configuration helpers, and a Development UI script.
- `tests/`: unit tests that validate the orchestration logic using a deterministic fake LLM.

## Requirements

1. Python 3.11+
2. [Ollama](https://ollama.ai) running locally with the `gemma3:12b` model pulled (`ollama pull gemma3:12b`).
3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Development UI

Start your local Ollama runtime and then execute:

```bash
python -m orchestrator_app.dev_ui
```

Enter free-form tasks. The orchestrator will ask Gemma3 to choose the best specialist and then display both the routing decision and the delegated agent's response. Type `exit` to quit.

## Programmatic usage

```python
from orchestrator_app import build_default_orchestrator
from google.adk import AgentRuntime

orchestrator = build_default_orchestrator()
runtime = AgentRuntime(orchestrator)
reply = runtime.send_user_message("Plan regression tests for the new parser")
print(reply.content)
```

## Testing

A deterministic fake LLM is provided to exercise the orchestration layer without requiring a running Ollama deployment:

```bash
pytest
```

The tests cover JSON-based routing, heuristic fallbacks, and ensure the orchestrator bundles delegated agent responses.

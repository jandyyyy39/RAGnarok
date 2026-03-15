# RAGnarok: Progressive Disclosure State Graph

**Abstract:** This document outlines an experimental multi-agent architecture designed to serve as the technical foundation for an academic study. The RAGnarok State Graph aims to mitigate common issues in monolithic LLM systems, specifically "Agent Bloat," hallucination, and high latency. It achieves this through a model-driven, dynamic routing system and a cyclic reflection loop, enabling more efficient, context-aware, and accurate agentic behavior compared to a traditional sequential pipeline.

## Architecture: Pipeline vs. Graph (Key Differences)

The core innovation of this experimental branch is the shift from a rigid, linear pipeline to a flexible, state-driven graph.

| Feature | Baseline: Sequential Prompt Chaining | Experimental: Model-Driven Decision-Making |
| :--- | :--- | :--- |
| **Control Flow** | Every input is forced through a static, monolithic chain of agents, including a heavy RAG Arbiter, regardless of intent. | A lightweight **SupervisorAgent** makes a dynamic routing decision at runtime, bypassing unnecessary nodes. |
| **Correction Loop**| No internal correction mechanism. Errors and hallucinations are passed directly to the user. | Implements a **ReAct (Reason -> Act -> Observe -> Reflect)** loop via a `Critic-Evaluator` node, which programmatically intercepts drafts, checks for constraint violations, and forces rewrites *before* a final response is generated. |
| **Context** | All context is loaded into a single, massive prompt, leading to bloat and potential performance degradation. | Context is managed via "Progressive Disclosure," injecting only the necessary information at the relevant step. |

## The "Progressive Disclosure" Agent Skills

To starve the model of unnecessary information that can lead to hallucinations, this architecture uses a filesystem-based skill library. Each skill is isolated in its own directory and exposes its capabilities through a three-tiered structure, ensuring minimal token overhead during the critical routing phase.

- **Level 1 (Metadata):** A simple `SKILL.md` file containing only the skill's name and a high-level, two-sentence description. This is the only information loaded by the `SupervisorAgent` at startup for routing purposes, typically consuming less than 100 tokens.

- **Level 2 (Instructions):** The full `SKILL.md` also contains detailed system prompts, operational workflows, constraints, and examples. This "just-in-time" context is dynamically injected into a worker agent *only after* the Supervisor has selected and activated the skill.

- **Level 3 (External Resources):** Each skill contains a `scripts/` subdirectory housing isolated Python scripts or other tools. These external resources are executed by the worker agent to interact with the outside world (e.g., databases, APIs).

### Active Skills

The current implementation includes two active skills:

1.  **`combat_mechanics`**: Triggered for actions related to fights, challenges, and physical interactions. Its Level 3 resource is a script that executes a vector search against a ChromaDB instance of game rules.
2.  **`social_lore`**: Triggered for dialogue, character inquiries, and narrative exploration. Its Level 3 resource is a script that performs a simple key-value lookup from an NPC profiles JSON file.

## The Node Roster (New Agents)

The state graph is composed of specialized agents, each with a single responsibility.

- **`SupervisorAgent` (Router):** The entry point of the graph. This agent is a strict, JSON-enforced router that evaluates the user's intent against the Level 1 Metadata of all available skills. It uses Chain-of-Thought (CoT) reasoning to produce a structured output declaring which skill (and corresponding execution path) to take.

- **`DMGraphAgent` (Worker):** A modular Dungeon Master agent. Unlike its monolithic counterpart in the baseline, this agent is a blank slate that accepts dynamically injected Level 2 instructions and the current world state. Its role is to narrate the game based on the activated skill and trigger UI events like dice rolls.

- **`Critic-Evaluator` (Guardrail):** A programmatic, non-LLM node that acts as a quality gate. It intercepts the `draft_narrative` from the `DMGraphAgent` and uses a series of regex checks to scan for meta-game terminology (e.g., `"DC"`, `"Skill Check"`, `"Roll"`). If any forbidden terms are found, it rejects the draft, populates an error feedback field, and forces the `DMGraphAgent` to perform a rewrite, effectively creating a self-correction loop.

## Telemetry & Study Metrics

To facilitate the academic A/B test, the graph orchestrator logs a structured JSON object to `data/history/experiment_log.json` on every turn. The baseline orchestrator does not generate this log.

The following metrics are captured for each interaction:

- **`latency_ms`**: An integer representing the total execution time in milliseconds from receiving the user input to dispatching the final response. This is used to prove the Supervisor's ability to bypass heavy RAG lookups for non-combat tasks.
- **`total_tokens`**: An integer representing the estimated sum of prompt and completion tokens used by all agents during the turn. This is used to prove that the Progressive Disclosure model uses fewer tokens than a single monolithic prompt.
- **`critic_rejections`**: An integer counting the number of times the `Critic-Evaluator` rejected the DM's draft during the turn. This tracks the self-correction error rate.
- **`route_taken`**: A string indicating the exact skill or path selected by the `SupervisorAgent` (e.g., `skills/combat_mechanics` or `skills/social_lore`). This tracks the routing accuracy of the Supervisor.

## Execution Instructions

The experimental backend is run using the `orchestrator_graph.py` entry point.

### Cloud/Groq LPUs
To run the system using a cloud-hosted model endpoint (e.g., Groq):
```bash
python orchestrator_graph.py
```

### Local Hardware (Ollama)
To run the system using a local model served via Ollama:
```bash
python orchestrator_graph.py --local
```

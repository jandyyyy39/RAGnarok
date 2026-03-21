# The SPEAR Architecture: A Technical Breakdown

This document details the architectural evolution of the RAGnarok TTRPG engine, transitioning from a monolithic, linear **Baseline** system to a modular, parallel **SPEAR** architecture.

## The Baseline (Monolithic/Linear)

The original Baseline architecture operates as a simple, sequential chain of agents. Every player input is processed through the exact same series of steps, regardless of context or intent.

### Structure

```mermaid
graph TD
    A[Player Input] --> B(Safety Agent);
    B --> C(Memory Agent);
    C --> D(Rules Arbiter);
    D --> E(Dungeon Master);
    E --> F(NPC Consistency Agent);
    F --> G[Final Output];
    F --> H{Async Critic};
```

### Workflow

The data flow is rigid. A player's input string is passed from one agent to the next, with each agent adding its analysis or transformation to the context before passing it downstream.

1.  **Player Input**: The initial turn action.
2.  **Safety Agent**: A basic filter for harmful content.
3.  **Memory Agent**: The full world state and recent events are injected into the prompt.
4.  **Rules Arbiter**: A vector search is performed against the game's rulebook.
5.  **Dungeon Master**: The LLM generates a response based on the combined context.
6.  **NPC Consistency Agent**: A final LLM call attempts to refine dialogue to match character profiles.
7.  **Async Reflection**: After the response is sent to the player, a non-blocking background thread sends the turn data to a **Critic Agent**. This agent evaluates the turn for rule accuracy and narrative coherence *without making the player wait*. Its findings are logged for analysis.

### Architectural Flaws

This design suffers from several critical inefficiencies:

-   **Agent Bloat**: Every agent in the primary chain (Safety, Memory, Rules, DM, NPC) fires on every single turn. A simple social interaction ("I ask Thrain for a drink") incorrectly triggers a time-consuming RAG search of the combat rulebook, wasting tokens and adding latency.
-   **Latency Stacking**: Because each step in the synchronous chain is a blocking API call that must complete before the next can begin, the total turn time is the sum of all individual agent latencies.
-   **The "Alzheimer's Effect"**: By injecting all available context (rules, history, NPCs) into a single monolithic prompt, the model's attention is diluted. This often leads to a degradation of persona and a failure to adhere to specific instructions as the token count climbs.

---

## The SPEAR Architecture (Modular/Parallel)

**SPEAR** (**S**upervised, **P**arallel, **E**valuated **A**gentic **R**outer) is a complete redesign focused on efficiency, modularity, and intelligent context management. It operates as a Progressive Disclosure State Machine, ensuring that only necessary resources are activated.

### Structure

```mermaid
graph TD
    subgraph SPEAR Architecture
        A[Player Input] --> R{Semantic Router};
        R -->|Intent: rules_logic| S1[Skill: Rules Logic];
        R -->|Intent: npc_lore| S2[Skill: NPC Lore];
        R -->|Intent: world_exploration| S3[Skill: World Exploration];

        subgraph "Parallel Skills (Data Fetch)"
            S1 --> P1((ChromaDB RAG));
            S2 --> P2((JSON Profile KV Lookup));
            S3 --> P1;
        end

        subgraph "Execution"
            P1 --> DM(DMSpearAgent);
            P2 --> DM;
            P3((No Context Needed)) --> DM;
            R -->|No Skill Match| P3;
        end

        DM --> E{Async Critic};
        DM --> F[Final Output];
    end
```

### Workflow

The SPEAR workflow is dynamic and intent-driven.

1.  **Semantic Router (Supervisor)**: The player's input is first sent to a lightweight, high-speed LLM router. This router's *only job* is to determine the user's intent by analyzing the query against a small set of "Skill" metadata.
2.  **Parallel Skills**: Based on the router's decision, the system fires off concurrent, non-blocking calls to fetch *only* the necessary data. If the intent is social, it queries the NPC JSON files. If it's combat-related, it triggers the ChromaDB vector search. These run in parallel.
3.  **Execution**: The `DMSpearAgent` is decoupled from the underlying mechanics. It receives a "payload" containing only the relevant instructions and retrieved context for the turn. This isolates the creative narrative task from the technical data-retrieval, preserving the LLM's persona and focus.
4.  **Async Reflection**: After the response is sent to the player, a non-blocking background thread sends the turn data to a **Critic Agent**. This agent evaluates the turn for rule accuracy and narrative coherence *without making the player wait*. Its findings are logged for analysis.

---

## Evaluation

The following models were used for evaluation of the two architectures.

| Role | Model | Architecture |
| :--- | :--- | :--- |
| **Dungeon Master** | `mistral-small3.2:24b` | Baseline & SPEAR |
| **Critic** | `llama3.1:8b` | Baseline & SPEAR |
| **Router** | `llama3.2:3b` | SPEAR Only |
| **Router** | `phi3.5:3.8b` | SPEAR Only |
| **Router** | `llama3.1:8b` | SPEAR Only |

---

## Technical Conclusion

While the Baseline architecture is simple to implement, its monolithic nature makes it fundamentally unscalable for real-time applications. The "Agent Bloat" and "Latency Stacking" create a poor user experience and incur unnecessarily high token costs, as every agent must be consulted on every turn.

**SPEAR is academically superior for real-time applications** because it treats latency and token cost as first-class constraints. By:
1.  Using a fast, intelligent **Semantic Router** to prune unnecessary work.
2.  Executing data retrieval in **parallel**.
3.  Isolating the narrative agent from mechanical details via **Persona Sandwiching**.
4.  Moving evaluation to a non-blocking **Async Reflection** loop.

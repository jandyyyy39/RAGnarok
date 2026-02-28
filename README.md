```
# Create the environment from the yaml
conda env create -f environment.yaml

# Activate the environment
conda activate ragnarok

# Fix for the Groq/HTTPX telemetry bug (Mandatory)
pip install httpx==0.27.2
```

# Progress & Architecture

## We have successfully moved from a monolithic script to a Modular Agent Pipeline. Each agent resides in agents/ and handles a specific cognitive task.
Current Features:  
* The Orchestrator (orchestrator.py): The central nervous system that routes data between specialized agents.  
* Rules Arbiter (RAG): Uses a ChromaDB vector store to query the official 5e SRD. It interprets player actions based on actual game mechanics rather than "best guesses."  
* Stateful Memory: Implemented a persistent world_state.json via the Memory Agent to track player HP, location, and NPC interactions.  
* Safety Guardrails: A dedicated Safety Agent that intercepts and blocks non-compliant content before it reaches the LLM.  
* Standardized Config: All global variables, model temperatures, and file paths are managed via a central config.py.

## Current Pipeline Flow:  
* Safety Check → Validate input.  
* Memory Recall → Inject world state context.  
* Rule Retrieval → Query RAG for mechanical DC and rolls.  
* Narrative Generation → DM Agent weaves story and mechanics.  
* Flavor Pass → NPC Consistency Agent applies dialogue quirks.  
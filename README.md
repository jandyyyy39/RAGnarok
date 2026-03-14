# RAGnarok

## Configuration & API Setup

Before running the orchestrator, you must set up your local environment variables.  
* ## Groq API Key: * Create a free account from https://console.groq.com/home.  
  - Generate a new API key and copy it.  
* ## Environment File:  
  - Create a file named .env in the root directory (this is hidden by .gitignore).  
  - Add the following lines:  
```
GROQ_API_KEY=your_key_here
ANONYMIZED_TELEMETRY=False
```  
### What is ANONYMIZED_TELEMETRY=False?  
We use this environment variable to disable ChromaDB's built-in usage tracking.  
* The Problem: By default, ChromaDB tries to send anonymous usage data to its servers. However, a version mismatch in its telemetry dependency (posthog) causes a "capture() error" that spams the console every time the Rules Arbiter queries the database.  
* The Fix: Setting this to False tells the database to skip the tracking attempt entirely. This stops the error messages and keeps our game logs clean and readable.

### Quick Start for Contributors

If you just cloned this repo, follow these steps to get the system running:
* Setup Conda: `conda env create -f environment.yaml` followed by `conda activate ragnarok`
* Initialize Database: Run `python scripts/build_rules_rag.py` to build your local ChromaDB vector store from the SRD.

#### Running the Application

**Backend (API Server):**
```bash
python orchestrator.py
```
This starts the Flask API server on `http://localhost:5000`.

### Command-line Arguments

The `orchestrator.py` script accepts the following command-line arguments for controlling its behavior:

*   `--local`: Use a local LLM model (Ollama) instead of the default Groq API.
*   `--no-rag`: Skip the Rules Arbiter agent. This is useful for ablation studies to see how the system behaves without the RAG component.
*   `--no-memory`: Prevents the Memory agent from injecting the world state into the prompt. This is for ablation studies to test the system's performance without memory.
*   `--no-npc-const`: Skips the NPC Consistency agent. This is for ablation studies to evaluate the impact of the consistency agent.

**Frontend (React UI):**
```bash
cd frontend
npm install    # First time only
npm run dev
```
This starts the Vite dev server (typically on `http://localhost:5173`).

Open the frontend URL in your browser to begin.

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

## TO DO:  
- [Shwe] D20 Dice rolling system (If AI responds with 'Roll a d20', we trigger a 'Roll dice' system)  
DM outputs dc15 -> DA -> True/False
"Rolled 15", "15"  
- [Andres] Campaign end + character progression + moving on to a different story (need Peimin)  
- [Andy] Clearer orchestration (thinking? visual?)  
- [Peimin+Lope] Quantitative evaluation  
- [Aditya] Fine-tuning and text-to-speech & speech-to-text  
- [Andres] History viewer  

Baseline:  
* Simple RAG  
* No finetuning for DM Agent  
* Simple Orchestration  

Eval metrics:  
* Perplexity  
* 

- [Peimin+Lope] Fine Tune -> Each person try different dataset  
- [Andres+Shwe] RAG -> Try more advanced shit  
- [Andy+Aditya] Architecture  -> Dig deeper into how it works  
- [Aditya] Input -> get clarity  
- [Everyone] Compare performance against baseline  
- [Andy] Research on GROQ with Adapters 
- [Andy] Message clarity (At the moment the reply is quite tedious, and feels 'boring' without someone else reading it out loud. How to make it better?)
- [Andy] Ablation study enabler  
Need to build an input -> output 'ground truth' database  

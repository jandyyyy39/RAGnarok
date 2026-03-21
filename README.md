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
  - The script automatically uses `data/5esrd.md` (already included in the repo).
  - No internet connection or HuggingFace token required for this step.

#### Running the Application

**Backend (API Server):**
```bash
python orchestrator.py
```
This starts the Flask API server on `http://localhost:5000`.

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
Baseline:  
* Simple RAG  
* No finetuning for DM Agent  
* Simple Orchestration  

Eval metrics:  
* Perplexity  

- [Peimin+Lope] Fine Tune -> Each person try different dataset  
- [Aditya] Input -> get clarity  
- [Everyone] Compare performance against baseline  

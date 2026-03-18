# RAGnarok — Fine-Tuning, Evaluation & Architecture Guide

This document explains the FIREBALL dataset, fine-tuning pipeline, hyperparameter grid search, evaluation system, and architecture changes. It is intended to help collaborators understand and extend the project.

---

## Table of Contents

1. [FIREBALL Dataset](#part-1--fireball-dataset)
2. [Fine-Tuning Logic](#part-2--fine-tuning-logic)
3. [Hyperparameter Grid Search](#part-3--hyperparameter-grid-search)
4. [Evaluation Logic](#part-4--evaluation-logic)
5. [Architecture Changes](#part-5--architecture-changes)
6. [How Fine-Tuning and Evaluation Connect](#part-6--how-fine-tuning-and-evaluation-connect)
7. [Commands Reference](#commands-reference)

---

## Part 1 — FIREBALL Dataset

### What Is FIREBALL?

FIREBALL is a dataset of **real recorded D&D 5e sessions**. Each row represents one moment in a game:

- **before_utterances**: What the player(s) said (e.g. "I draw my sword and charge the goblin")
- **after_utterances**: What the DM said in response (e.g. "The goblin snarls as your blade connects. Roll for damage.")

The dataset contains roughly 25,000 such player→DM pairs from actual gameplay.

---

### Scripts and Flow

| Script | Purpose |
|--------|---------|
| `download_fireball.py` | **Inspection only.** Checks for local FIREBALL files, prints schema and sample rows. Does not modify data. |
| `prepare_fireball.py` | **Transformation.** Loads raw data, extracts instruction/input/output triples, splits 90/10, saves `fireball_train.jsonl` and `fireball_eval.jsonl`. |

**Data sources (in order of precedence):**
1. Local files in `data/` or `data/fireball/` — `.parquet`, `.csv`, or `.jsonl` with "fireball" in the filename
2. HuggingFace Hub (`lara-martin/FIREBALL`) — requires `HF_TOKEN` if the dataset is gated

---

### How `prepare_fireball.py` Works

1. **Find data** — Scans `data/`, `data/fireball/`, `data/FIREBALL/` for matching files
2. **Load** — Reads parquet, CSV, or JSONL depending on file type
3. **Extract** — For each row:
   - `before_utterances` → joined into `input` (player context)
   - `after_utterances` → joined into `output` (DM narration)
   - Adds fixed `instruction` (system prompt)
4. **Filter** — Drops rows where `input` < 10 chars or `output` < 20 chars
5. **Split** — Shuffles with `random.seed(42)`, 90% train / 10% eval
6. **Save** — Writes `fireball_train.jsonl` and `fireball_eval.jsonl`

---

### Output Format (per line in JSONL)

```json
{
  "instruction": "You are an expert Dungeon Master running a D&D 5e campaign...",
  "input": "I draw my sword and charge the goblin",
  "output": "The goblin snarls as your blade connects. Roll for damage."
}
```

---

### Which Agent Uses FIREBALL?

FIREBALL is used **only for the DM Agent**. The other agents do not use it:

| Agent | Role | Uses FIREBALL? |
|-------|------|----------------|
| Safety Agent | Blocks forbidden inputs | No |
| Memory Agent | Tracks world state | No |
| Rules Arbiter | Queries SRD rules | No |
| **DM Agent** | Generates narrative | **Yes** |
| NPC Consistency Agent | Rewrites dialogue to match NPC profiles | No |

---

## Part 2 — Fine-Tuning Logic

### What We're Teaching

The **DM Agent** is the only component that generates narrative. It must turn a player action into a Dungeon Master–style response. The FIREBALL dataset provides real examples of this:

```
Player: "I draw my sword and charge the goblin"
DM:     "The goblin snarls as your blade connects. Roll for damage."
```

We train the model to imitate this mapping.

---

### The Training Format (Instruction Tuning)

Each FIREBALL row is converted into a chat-style triple:

```
System:    "You are an expert Dungeon Master running a D&D 5e campaign..."
User:      "I draw my sword and charge the goblin"
Assistant: "The goblin snarls as your blade connects. Roll for damage."
```

This matches the chat template that Llama 3.2 Instruct was pre-trained on. The model learns: *given this system prompt and user message, produce this assistant reply*.

---

### Why LoRA Instead of Full Fine-Tuning

- **Full fine-tuning**: Update all 3 billion parameters → requires ~24GB+ VRAM.
- **LoRA**: Freeze the base model, add small adapter matrices (A, B) in attention layers → only ~24M parameters trained → fits in ~9GB VRAM.

LoRA modifies *how* the model responds without retraining everything. For style and tone (DM narration), that's sufficient.

---

### The Two-Stage Process

**Stage 1 — Hyperparameter Search (`hparam_search.py`):**

- Uses a small subset (300 train, 75 eval samples).
- Trains 1 epoch per combination of rank, alpha, and learning rate.
- Measures **eval loss** (cross-entropy on held-out examples).
- Lower eval loss = better predictions on unseen DM narrations.
- Selects the combination with the lowest eval loss.

**Stage 2 — Full Fine-Tuning (`finetune_fireball.py`):**

- Uses the best rank, alpha, and lr from Stage 1.
- Trains on the full dataset for 3 epochs.
- Exports to GGUF for Ollama deployment.

---

### Why This Makes Sense

1. **Real data**: FIREBALL contains actual D&D sessions, not synthetic data.
2. **Efficiency**: Grid search on a subset finds good hyperparameters without 18 full training runs.
3. **Reproducibility**: Fixed seeds (42) and the same train/eval split ensure repeatable results.
4. **Hardware**: LoRA enables fine-tuning on consumer GPUs (e.g. RTX 4070).

---

## Part 3 — Hyperparameter Grid Search

### What It Does

`hparam_search.py` finds the best LoRA hyperparameters **before** running the full 3-epoch training. Instead of guessing rank, alpha, and learning rate, we systematically test combinations and pick the one with the lowest eval loss.

---

### The Three Parameters

| Parameter | What It Controls | Typical Values |
|-----------|------------------|----------------|
| **rank (r)** | Width of LoRA adapter matrices. Higher = more capacity to learn, but more VRAM. | 8, 16, 32 |
| **lora_alpha** | Scaling factor for adapter output. Often set equal to rank. | 8, 16, 32 |
| **learning_rate** | Step size during gradient descent. Too high = unstable; too low = no learning. | 5e-5, 1e-4, 2e-4 |

---

### Why Use a Subset?

- Full training: ~22,500 samples × 3 epochs ≈ 2–3 hours per combination
- 18 combinations × 3 hours = 54 hours
- **Subset approach**: 300 train / 75 eval × 1 epoch ≈ 10–40 min per combination
- 18 combinations × ~20 min ≈ 6 hours total

The subset approximates full-data performance. The **ranking** of configs (which is best) is usually preserved even on a small subset.

---

### How It Runs

1. Load base model **once** (frozen weights stay in VRAM)
2. For each (rank, alpha, lr) combination:
   - Attach new LoRA adapters with those params
   - Train 1 epoch on 300 samples
   - Evaluate on 75 held-out samples → record `eval_loss`
   - Delete adapters, clear GPU cache
3. Sort by eval loss (lower = better)
4. Print best config and the exact `finetune_fireball.py` command

---

### Interpreting Results

- **eval_loss**: Cross-entropy on the eval set. Lower = model predicts the held-out DM narrations more accurately.
- **Best config** is printed at the end. Use those values for full fine-tuning.

---

## Part 4 — Evaluation Logic

### What We're Measuring

We compare different system configurations (full pipeline vs. no RAG vs. fine-tuned local) on the same inputs and measure how close the outputs are to human DM narrations.

---

### Where Ground Truth Comes From

`fireball_eval.jsonl` is the 10% held-out split from `prepare_fireball.py`. Each row has:

- `input`: player action
- `output`: reference DM narration (human-written)

These samples are never used for training, so they serve as ground truth for evaluation.

---

### The Evaluation Flow

1. Load test inputs from `fireball_eval.jsonl` (or a fallback list if the file is missing).
2. For each experimental configuration (Full Pipeline, No RAG, Local Fine-tuned, etc.):
   - Instantiate the orchestrator with that config.
   - Run each test input through the system.
   - Capture the DM's response.
3. For each (input, system_response, reference) triple:
   - Compute metrics comparing `system_response` to `reference`.

---

### The Metrics (All Local, No API)

| Metric | What It Measures |
|--------|------------------|
| **ROUGE-1/2/L** | N-gram overlap with the reference. Higher = more similar wording. |
| **BERTScore** | Semantic similarity via a local BERT model. Higher = more similar meaning. |
| **Rule Coverage** | Fraction of D&D mechanical terms (roll, DC, check, etc.) present in the response. |
| **Latency** | End-to-end time per turn (ms). |
| **Response Length** | Word count of the response. |


---

### Why This Makes Sense

1. **Ground truth**: We compare against real human DM narrations, not arbitrary scores.
2. **Comparable conditions**: Same inputs and metrics for every configuration.
3. **Multiple dimensions**: ROUGE for surface similarity, BERTScore for semantics, rule coverage for mechanics, latency for speed.

---

## Part 5 — Architecture Changes

### Overview

The orchestrator was refactored to support **experimentation** — different configurations can be toggled via command-line flags for ablation studies and comparison.

**Default pipeline flow:**
```
Safety Check → Input Classifier → Memory Recall → Rules Arbiter → DM Agent → NPC Consistency → (optional TTS)
```

Agents can be disabled via flags for ablation. The Input Classifier can also skip RAG or NPC pass for certain input types when smart routing is enabled.

---

### Command-Line Flags

| Flag | Effect |
|------|--------|
| `--local` | Use Ollama (fine-tuned or base model) instead of Groq API |
| `--no-rag` | Disable Rules Arbiter — DM gets no mechanical ruling from SRD |
| `--no-memory` | Disable Memory Agent — DM gets no world state context |
| `--no-npc-const` | Disable NPC Consistency Agent — no dialogue flavour pass |
| `--voice` | Enable TTS (edge-tts) — speak DM narration aloud |
| `--no-smart-routing` | Disable input-based agent skipping (see below) |

---

### Input Classifier & Smart Routing

**`agents/input_classifier.py`** classifies every player input into a type:

| Type | Examples |
|------|----------|
| `dice_result` | "I rolled a 17", "Natural 20!" |
| `combat_action` | "I attack the goblin", "I cast Fireball" |
| `roleplay` | "I tell the guard...", "I persuade the merchant" |
| `exploration` | "I search the room", "I examine the door" |
| `rules_question` | "Can I cast two spells?", "How does grappling work?" |
| `out_of_game` | "pause", "help", "undo" |
| `general_action` | Fallback for anything else |

**Smart routing** uses this to skip agents when they add little value:

- **Skip RAG** for `roleplay`, `out_of_game`, `dice_result` — these rarely need SRD lookups
- **Skip NPC pass** for `out_of_game`, `dice_result` — no NPCs in the scene

Use `--no-smart-routing` to disable this and run the full pipeline for every turn (needed for fair ablation comparison).

---

### Experimental Configurations (in `evaluate.py`)

| Configuration | Flags | What It Tests |
|---------------|-------|---------------|
| Full Pipeline (Groq) | default | Baseline — all agents, cloud model |
| No RAG | `--no-rag` | Does rules grounding help? |
| No Memory | `--no-memory` | Does world state context help? |
| No NPC Pass | `--no-npc-const` | Does the consistency agent add value? |
| Bare DM Only | `--no-rag --no-memory --no-npc-const` | Floor baseline — DM alone |
| Local (Mistral) | `--local` | Local model vs. Groq |
| Local FT + No RAG | `--local --no-rag` | Does fine-tuning reduce RAG dependency? |

---

### DM Agent: Groq vs. Ollama

The DM Agent supports two backends via the `use_local` flag:

- **Groq**: `Config.LLM_MODEL['GROQ']` — e.g. `llama-3.3-70b-versatile`
- **Ollama**: `Config.LLM_MODEL['LOCAL']` — e.g. `mistral-small3.2` or fine-tuned `ragnarok-dm`

Both use the same OpenAI-compatible `chat.completions.create()` interface, so the rest of the code is unchanged.

---

### Per-Turn Metadata

Each `process_turn()` returns a dict with:

- `input`, `input_type`, `response`
- `safety_pass`, `ruling`, `dm_raw`
- `latency_ms`, `skipped` (list of agents skipped by smart routing)

This supports evaluation scripts and debugging.

---

## Part 6 — How Fine-Tuning and Evaluation Connect

```
FIREBALL dataset
    │
    ├── 90% → fireball_train.jsonl  →  fine-tuning (teach the model)
    │
    └── 10% → fireball_eval.jsonl   →  evaluation (measure quality)
              │
              └── Same file used by evaluate.py as ground truth references
```

The same held-out 10% is used to:

1. **During training**: Compute validation loss and select the best checkpoint.
2. **During evaluation**: Provide reference DM narrations for ROUGE/BERTScore.

We never train on the eval split, and both validation and final evaluation use the same unseen data. This keeps the evaluation fair and aligned with how the model was validated.

---

## Commands Reference

### Fine-Tuning Pipeline (WSL2)

```bash
# 1. Prepare the dataset
python scripts/prepare_fireball.py

# 2. Hyperparameter search (use smaller grid for speed)
python scripts/hparam_search.py --ranks 8 16 --lrs 5e-5 1e-4 2e-4

# 3. Full fine-tuning with best params (from search output)
python scripts/finetune_fireball.py --rank 16 --alpha 32 --lr 0.0002

# 4. Register in Ollama
ollama create ragnarok-dm -f data/ragnarok-dm-gguf/Modelfile
ollama run ragnarok-dm
```

### Evaluation (No API Required)

```bash
# Quick smoke test (8 inputs)
python scripts/evaluate.py --quick

# Full evaluation (25 inputs × all configs)
python scripts/evaluate.py

# Re-generate plots from existing results
python scripts/evaluate.py --plot-only
```

### Run System with Fine-Tuned Model

```bash
python orchestrator.py --local
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/fireball_train.jsonl` | Training samples (90% of FIREBALL) |
| `data/fireball_eval.jsonl` | Eval samples (10%) — used for validation and evaluation |
| `data/hparam_results.json` | Hyperparameter search results |
| `data/ragnarok-dm-gguf/` | Exported fine-tuned model for Ollama |
| `data/eval_results.json` | Raw evaluation scores per run |
| `data/plots/*.png` | Comparison charts (ROUGE, latency, heatmap, etc.) |

---

## End-to-End Pipeline (Script Dependencies)

```
download_fireball.py     (optional — inspect schema first)
        │
        ▼
prepare_fireball.py      → fireball_train.jsonl, fireball_eval.jsonl
        │
        ▼
hparam_search.py         → hparam_results.json, prints best rank/alpha/lr
        │
        ▼
finetune_fireball.py     → ragnarok-dm-gguf/, Modelfile
        │
        ▼
ollama create ragnarok-dm
        │
        ▼
orchestrator.py --local  (uses fine-tuned model)
        │
        ▼
evaluate.py              (compares all configs, uses fireball_eval.jsonl as ground truth)
```

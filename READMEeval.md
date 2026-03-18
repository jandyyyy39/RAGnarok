# RAGnarok — Fine-Tuning & Evaluation Guide

This document explains how the FIREBALL fine-tuning pipeline and evaluation system work, and why the design choices make sense.

---

## Table of Contents

1. [Fine-Tuning Logic](#part-1--fine-tuning-logic)
2. [Evaluation Logic](#part-2--evaluation-logic)
3. [How They Connect](#part-3--how-fine-tuning-and-evaluation-connect)
4. [Commands Reference](#commands-reference)

---

## Part 1 — Fine-Tuning Logic

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

## Part 2 — Evaluation Logic

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

## Part 3 — How Fine-Tuning and Evaluation Connect

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

## Part 4 — Which Agent Uses FIREBALL?

FIREBALL is used **only for the DM Agent**. The other agents do not use it:

| Agent | Role | Uses FIREBALL? |
|-------|------|----------------|
| Safety Agent | Blocks forbidden inputs | No |
| Memory Agent | Tracks world state | No |
| Rules Arbiter | Queries SRD rules | No |
| **DM Agent** | Generates narrative | **Yes** |
| NPC Consistency Agent | Rewrites dialogue to match NPC profiles | No |

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

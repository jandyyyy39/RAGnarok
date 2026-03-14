"""
prepare_fireball.py
-------------------
Downloads the FIREBALL D&D dataset and converts it into an instruction-tuning
format ready for QLoRA fine-tuning with Unsloth.

Output: data/fireball_train.jsonl  (instruction / input / output triples)
        data/fireball_eval.jsonl   (held-out 10% for validation)

Fine-tuning target: DMAgent — we teach the model to generate immersive DM
narrations given a player action + world context.

Usage:
    python scripts/prepare_fireball.py

After this script, run fine-tuning (WSL2 recommended on Windows):
    pip install unsloth
    python scripts/finetune_fireball.py
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "fireball_train.jsonl"
EVAL_FILE  = DATA_DIR / "fireball_eval.jsonl"

SYSTEM_PROMPT = (
    "You are an expert Dungeon Master running a D&D 5e campaign. "
    "Given a player's action and the current world state, generate an immersive, "
    "mechanically-grounded narrative response. Keep the tone engaging and vivid."
)


def extract_samples(dataset) -> list[dict]:
    """
    FIREBALL schema has these relevant fields per turn:
      - before_utterances : list of strings (context / player speech before DM turn)
      - after_utterances  : list of strings (DM narration)
      - commands_norm     : normalized dice/action commands the DM issued

    We build instruction samples of the form:
      instruction : system prompt
      input       : player action(s) joined as context
      output      : DM narration
    """
    samples = []
    split = dataset.get("train", dataset[list(dataset.keys())[0]])

    for row in split:
        before = row.get("before_utterances") or []
        after  = row.get("after_utterances")  or []

        if not before or not after:
            continue

        player_context = " ".join(str(u) for u in before).strip()
        dm_response    = " ".join(str(u) for u in after).strip()

        if len(player_context) < 10 or len(dm_response) < 20:
            continue

        samples.append({
            "instruction": SYSTEM_PROMPT,
            "input":       player_context,
            "output":      dm_response,
        })

    return samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples):,} samples → {path}")


def main():
    print("Loading FIREBALL dataset from Hugging Face Hub...")
    dataset = load_dataset("lara-martin/FIREBALL", trust_remote_code=True)
    print(f"  Dataset keys: {list(dataset.keys())}")

    print("Extracting DM narration samples...")
    samples = extract_samples(dataset)
    print(f"  Extracted {len(samples):,} usable samples")

    if not samples:
        print("ERROR: No samples extracted. Check FIREBALL schema — field names may have changed.")
        return

    # Shuffle and split 90/10
    random.seed(42)
    random.shuffle(samples)
    split_idx   = int(len(samples) * 0.9)
    train_data  = samples[:split_idx]
    eval_data   = samples[split_idx:]

    print("Saving splits...")
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(eval_data,  EVAL_FILE)

    print("\nDataset ready for fine-tuning.")
    print("Next step: run scripts/finetune_fireball.py (in WSL2 with Unsloth installed)")
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Eval : {len(eval_data):,} samples")


if __name__ == "__main__":
    main()

"""
prepare_cr3d.py
---------------
Extracts DM narration training pairs from the CRD3 dataset
([CRD3 on GitHub](https://github.com/RevanthRameshkumar/CRD3/tree/master/data)).
Uses official `train_files` / `val_files` splits under `data/aligned data/`.

Input:  data/crd3/data/aligned data/c=2/*.json  (+ train_files, val_files)
Output: data/crd3_train.jsonl, data/crd3_eval.jsonl

Usage:
    python scripts/prepare_cr3d.py
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config

ALIGNED_DIR = Config.BASE_DIR / "data" / "crd3" / "data" / "aligned data"
CRD3_DIR    = ALIGNED_DIR / "c=2"
OUTPUT_DIR  = Config.DATA_DIR
TRAIN_FILE  = OUTPUT_DIR / "crd3_train.jsonl"
EVAL_FILE   = OUTPUT_DIR / "crd3_eval.jsonl"

SYSTEM_PROMPT = (
    "You are an expert Dungeon Master running a D&D 5e campaign. "
    "Given the players' actions and dialogue, generate an immersive, dramatic narrative "
    "response. Stay in character. Keep the tone vivid and engaging."
)

OOC_PATTERNS = re.compile(
    r"\b(geek.?sundry|twitch|live.?stream|bathroom break|we.ll be right back"
    r"|follow us on|subscribe|twitter|instagram|roll.?initiative for the show"
    r"|that.s our show|thanks for watching|see you next week)\b",
    re.IGNORECASE
)

MIN_PLAYER_LEN = 20
MIN_DM_LEN     = 80


def load_split_keys(filename: str) -> set[str]:
    path = ALIGNED_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_episode_turns(json_path: Path) -> dict[int, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    turns = {}
    for chunk in chunks:
        for turn in chunk.get("TURNS", []):
            num = turn["NUMBER"]
            if num not in turns:
                turns[num] = {
                    "names":      turn.get("NAMES", []),
                    "utterances": turn.get("UTTERANCES", []),
                }
    return turns


def is_matt_turn(turn: dict) -> bool:
    return any(n.upper() == "MATT" for n in turn["names"])


def join_utterances(utterances: list[str]) -> str:
    return " ".join(u.strip() for u in utterances if u.strip())


def extract_pairs_from_episode(turns: dict[int, dict]) -> list[dict]:
    pairs = []
    player_buffer = []

    for num in sorted(turns.keys()):
        turn = turns[num]
        text = join_utterances(turn["utterances"])

        if not text:
            continue

        if is_matt_turn(turn):
            if OOC_PATTERNS.search(text):
                player_buffer = []
                continue
            if len(text) < MIN_DM_LEN:
                player_buffer = []
                continue
            if player_buffer:
                player_context = " | ".join(player_buffer)
                if len(player_context) >= MIN_PLAYER_LEN:
                    pairs.append({
                        "instruction": SYSTEM_PROMPT,
                        "input":       player_context,
                        "output":      text,
                    })
            player_buffer = []
        else:
            player_buffer.append(text)
            if len(player_buffer) > 5:
                player_buffer.pop(0)

    return pairs


def process_split(episode_keys: set[str], json_files: list[Path]) -> list[dict]:
    # Group files by episode key
    episodes = defaultdict(list)
    for f in json_files:
        episode_key = f.stem.split("_")[0]
        if episode_key in episode_keys:
            episodes[episode_key].append(f)

    all_pairs = []
    for episode_key, files in sorted(episodes.items()):
        merged_turns = {}
        for f in files:
            merged_turns.update(load_episode_turns(f))
        all_pairs.extend(extract_pairs_from_episode(merged_turns))

    return all_pairs


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples):,} samples → {path}")


def main():
    if not CRD3_DIR.exists():
        print(f"ERROR: CRD3 data not found at {CRD3_DIR}")
        print("From repo root:")
        print('  git clone https://github.com/RevanthRameshkumar/CRD3 "data/crd3"')
        print("Then confirm: data/crd3/data/aligned data/c=2/*.json exists.")
        return

    # Load official splits
    train_keys = load_split_keys("train_files")
    val_keys   = load_split_keys("val_files")
    print(f"Official split: {len(train_keys)} train episodes, {len(val_keys)} val episodes")

    json_files = sorted(CRD3_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in c=2/")

    print("Processing train episodes...")
    train_data = process_split(train_keys, json_files)
    random.seed(42)
    random.shuffle(train_data)

    print("Processing val episodes...")
    eval_data = process_split(val_keys, json_files)
    random.seed(42)
    random.shuffle(eval_data)

    print(f"\nExtracted {len(train_data):,} train pairs, {len(eval_data):,} eval pairs")

    if not train_data:
        print("ERROR: No pairs extracted. Check CRD3 data structure.")
        return

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(eval_data,  EVAL_FILE)

    print(f"\nSample pair:")
    print(f"  INPUT:  {train_data[0]['input'][:120]}...")
    print(f"  OUTPUT: {train_data[0]['output'][:120]}...")
    print("\nNext step (CRD3 fine-tune, separate run name — does not overwrite ragnarok-dm):")
    print('  python scripts/finetune_fireball.py --train-file data/crd3_train.jsonl --eval-file data/crd3_eval.jsonl \\')
    print('    --run-name crd3-llama32-r16a32-lr2e4 --rank 16 --alpha 32 --lr 0.0002')


if __name__ == "__main__":
    main()

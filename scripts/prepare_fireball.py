import json
import random
from pathlib import Path
from config import Config

DATA_DIR   = Config.DATA_DIR
TRAIN_FILE = Config.TRAIN_FILE
EVAL_FILE  = Config.EVAL_FILE

SYSTEM_PROMPT = (
    "You are an expert Dungeon Master running a D&D 5e campaign. "
    "Given a player's action and the current world state, generate an immersive, "
    "mechanically-grounded narrative response. Keep the tone engaging and vivid."
)


def find_local_fireball() -> list[Path]:
    patterns = ["*.parquet", "*.csv", "*.jsonl", "*.json"]
    candidates = []
    search_dirs = [DATA_DIR, DATA_DIR / "fireball", DATA_DIR / "FIREBALL"]
    for d in search_dirs:
        if d.exists():
            for pat in patterns:
                candidates.extend(d.glob(pat))
    return [f for f in candidates if "fireball" in f.name.lower() or f.parent.name.lower() == "fireball"]


def load_parquet(files: list[Path]) -> list[dict]:
    import pandas as pd
    frames = [pd.read_parquet(f) for f in files]
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded parquet: {len(df):,} rows, columns: {list(df.columns)}")
    return df.to_dict(orient="records")


def load_csv(files: list[Path]) -> list[dict]:
    import pandas as pd
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded CSV: {len(df):,} rows")
    return df.to_dict(orient="records")


def load_jsonl_files(files: list[Path]) -> list[dict]:
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    print(f"  Loaded JSONL: {len(rows):,} rows")
    return rows


def load_from_hf() -> list[dict]:
    import os
    from datasets import load_dataset
    token = Config.HF_TOKEN or os.getenv("HF_TOKEN")
    print("Attempting HuggingFace download (lara-martin/FIREBALL)...")
    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    dataset = load_dataset("lara-martin/FIREBALL", **kwargs)
    split   = dataset.get("train", dataset[list(dataset.keys())[0]])
    print(f"  Downloaded {len(split):,} rows from HuggingFace")
    return [dict(row) for row in split]


def extract_samples(rows: list[dict]) -> list[dict]:
    samples = []
    for row in rows:
        before = row.get("before_utterances") or []
        after  = row.get("after_utterances")  or []

        if isinstance(before, str):
            before = [before]
        if isinstance(after, str):
            after = [after]

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
    local_files = find_local_fireball()

    if local_files:
        parquet = [f for f in local_files if f.suffix == ".parquet"]
        csvs    = [f for f in local_files if f.suffix == ".csv"]
        jsonls  = [f for f in local_files if f.suffix in (".jsonl", ".json")]

        print(f"Found {len(local_files)} local FIREBALL file(s):")
        for f in local_files:
            print(f"  {f}")

        if parquet:
            rows = load_parquet(parquet)
        elif csvs:
            rows = load_csv(csvs)
        elif jsonls:
            rows = load_jsonl_files(jsonls)
        else:
            rows = []
    else:
        print("No local FIREBALL files found. Trying HuggingFace...")
        try:
            rows = load_from_hf()
        except Exception as e:
            print(f"ERROR: Could not load FIREBALL data. {e}")
            print("\nTo use local files, place FIREBALL data files in one of:")
            print(f"  {DATA_DIR}/          (any .parquet, .csv, or .jsonl with 'fireball' in name)")
            print(f"  {DATA_DIR}/fireball/ (any .parquet, .csv, or .jsonl)")
            return

    if not rows:
        print("ERROR: No data rows loaded.")
        return

    print(f"Extracting DM narration samples from {len(rows):,} rows...")
    samples = extract_samples(rows)
    print(f"  {len(samples):,} usable samples extracted")

    if not samples:
        print("ERROR: No samples extracted. The data may use different column names.")
        print(f"  Available columns: {list(rows[0].keys()) if rows else 'unknown'}")
        return

    random.seed(42)
    random.shuffle(samples)
    split_idx  = int(len(samples) * 0.9)
    train_data = samples[:split_idx]
    eval_data  = samples[split_idx:]

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(eval_data,  EVAL_FILE)

    print(f"\nTrain: {len(train_data):,}  Eval: {len(eval_data):,}")
    print("Run fine-tuning next: python scripts/hparam_search.py")


if __name__ == "__main__":
    main()

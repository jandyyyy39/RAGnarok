import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from scripts.prepare_fireball import find_local_fireball, load_from_hf


def inspect_schema(rows: list[dict]):
    if not rows:
        print("No rows to inspect.")
        return
    print(f"\nTotal rows: {len(rows):,}")
    print(f"Columns   : {list(rows[0].keys())}")
    print("\nSample row:")
    sample = rows[0]
    for k, v in sample.items():
        preview = str(v)[:120]
        print(f"  {k:30s}: {preview}")


def main():
    local = find_local_fireball()
    if local:
        print(f"Local FIREBALL files found:")
        for f in local:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}  ({size_mb:.1f} MB)")

        parquet = [f for f in local if f.suffix == ".parquet"]
        if parquet:
            from scripts.prepare_fireball import load_parquet
            rows = load_parquet(parquet)
            inspect_schema(rows)
        return

    print("No local files found. Downloading from HuggingFace...")
    try:
        rows = load_from_hf()
        inspect_schema(rows)
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nPlace FIREBALL data files (.parquet / .csv / .jsonl) in:")
        print(f"  {Config.DATA_DIR}/fireball/")


if __name__ == "__main__":
    main()

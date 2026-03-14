"""
finetune_fireball.py
--------------------
QLoRA fine-tuning of Llama 3.2 3B on FIREBALL DM narrations using Unsloth.

Requirements (run inside WSL2 Ubuntu on your RTX 4070):
    pip install unsloth
    pip install torch --index-url https://download.pytorch.org/whl/cu121

Your RTX 4070 has 12 GB VRAM — Llama 3.2 3B at 4-bit needs ~6 GB, comfortable.

After training, the script exports a GGUF file for Ollama and prints
the commands to register it as the "ragnarok-dm" model.

Usage:
    python scripts/finetune_fireball.py
    python scripts/finetune_fireball.py --model llama3.2-1b  # lighter option
"""

import argparse
import json
from pathlib import Path

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "fireball_train.jsonl"
EVAL_FILE  = DATA_DIR / "fireball_eval.jsonl"
OUTPUT_DIR = DATA_DIR / "ragnarok-dm-lora"
GGUF_DIR   = DATA_DIR / "ragnarok-dm-gguf"

MODELS = {
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
}

MAX_SEQ_LEN  = 2048
LORA_RANK    = 16
TRAIN_EPOCHS = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 4       # effective batch = 8


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(sample: dict, tokenizer) -> str:
    """Alpaca-style prompt that Unsloth's chat template understands."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": sample["instruction"]},
            {"role": "user",      "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2-3b", choices=list(MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer via Unsloth (4-bit QLoRA)
    # ------------------------------------------------------------------
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: Unsloth not found. Install with: pip install unsloth")
        print("This script must be run in WSL2 (Linux) with CUDA available.")
        return

    model_id = MODELS[args.model]
    print(f"Loading {model_id} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_id,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = True,
        dtype          = None,       # auto-detect bf16 / fp16
    )

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r                   = LORA_RANK,
        lora_alpha          = LORA_RANK,
        lora_dropout        = 0.05,
        target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        bias                = "none",
        use_gradient_checkpointing = "unsloth",
        random_state        = 42,
    )

    # ------------------------------------------------------------------
    # 3. Load and format dataset
    # ------------------------------------------------------------------
    if not TRAIN_FILE.exists():
        print(f"ERROR: {TRAIN_FILE} not found. Run prepare_fireball.py first.")
        return

    print("Loading FIREBALL splits...")
    train_raw = load_jsonl(TRAIN_FILE)
    eval_raw  = load_jsonl(EVAL_FILE)

    from datasets import Dataset
    train_ds = Dataset.from_list([{"text": format_prompt(s, tokenizer)} for s in train_raw])
    eval_ds  = Dataset.from_list([{"text": format_prompt(s, tokenizer)} for s in eval_raw])
    print(f"  Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    from trl import SFTTrainer
    from transformers import TrainingArguments

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        dataset_text_field = "text",
        max_seq_length  = MAX_SEQ_LEN,
        args = TrainingArguments(
            output_dir              = str(OUTPUT_DIR),
            num_train_epochs        = args.epochs,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps            = 50,
            learning_rate           = 2e-4,
            fp16                    = True,
            logging_steps           = 25,
            eval_strategy           = "epoch",
            save_strategy           = "epoch",
            load_best_model_at_end  = True,
            seed                    = 42,
            report_to               = "none",
        ),
    )

    print("\nStarting fine-tuning... (this will take ~2-3 hours on RTX 4070)")
    trainer.train()
    print("Training complete!")

    # ------------------------------------------------------------------
    # 5. Export to GGUF for Ollama
    # ------------------------------------------------------------------
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting to GGUF (q4_k_m quantisation) → {GGUF_DIR} ...")
    model.save_pretrained_gguf(
        str(GGUF_DIR),
        tokenizer,
        quantization_method = "q4_k_m",
    )

    # ------------------------------------------------------------------
    # 6. Print Ollama registration instructions
    # ------------------------------------------------------------------
    gguf_file = list(GGUF_DIR.glob("*.gguf"))
    gguf_path = gguf_file[0] if gguf_file else GGUF_DIR / "model.gguf"

    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_path.write_text(
        f'FROM {gguf_path}\n'
        'PARAMETER temperature 0.7\n'
        'SYSTEM "You are a master storyteller and Dungeon Master for a D&D 5e campaign."\n'
    )

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("\nTo register in Ollama, run:")
    print(f"  ollama create ragnarok-dm -f {modelfile_path}")
    print("\nTo test:")
    print("  ollama run ragnarok-dm")
    print("\nTo run RAGnarok with the fine-tuned model:")
    print("  python orchestrator.py --local")
    print("=" * 60)


if __name__ == "__main__":
    main()

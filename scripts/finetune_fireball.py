import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config

DATA_DIR   = Config.DATA_DIR
TRAIN_FILE = Config.TRAIN_FILE
EVAL_FILE  = Config.EVAL_FILE
OUTPUT_DIR = DATA_DIR / "ragnarok-dm-lora"
GGUF_DIR   = DATA_DIR / "ragnarok-dm-gguf"

MODELS = {
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
}

MAX_SEQ_LEN  = 1024   # Lower for 12GB VRAM; increase if you have more
BATCH_SIZE   = 1       # Reduce to avoid fused cross-entropy OOM on RTX 4070
GRAD_ACCUM   = 8       # Compensate: effective batch = 1 * 8 = 8


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(sample: dict, tokenizer) -> str:
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
    parser.add_argument("--model",  default="llama3.2-3b", choices=list(MODELS.keys()))
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--rank",   type=int,   default=16,   help="LoRA rank")
    parser.add_argument("--alpha",  type=int,   default=16,   help="LoRA alpha")
    parser.add_argument("--lr",     type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Per-device batch size (default 1 for 12GB VRAM)")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN, help="Max sequence length (default 1024)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading (needs ~24GB VRAM)")
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: run inside WSL2 with: pip install unsloth")
        return

    model_id = MODELS[args.model]
    print(f"Loading {model_id} | rank={args.rank} alpha={args.alpha} lr={args.lr}")
    max_seq = args.max_seq_len
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_id,
        max_seq_length = max_seq,
        load_in_4bit   = not args.no_4bit,
        dtype          = None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = args.rank,
        lora_alpha                 = args.alpha,
        lora_dropout               = 0.05,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )

    if not TRAIN_FILE.exists():
        print(f"ERROR: {TRAIN_FILE} not found. Run prepare_fireball.py first.")
        return

    train_raw = load_jsonl(TRAIN_FILE)
    eval_raw  = load_jsonl(EVAL_FILE)

    from datasets import Dataset
    train_ds = Dataset.from_list([{"text": format_prompt(s, tokenizer)} for s in train_raw])
    eval_ds  = Dataset.from_list([{"text": format_prompt(s, tokenizer)} for s in eval_raw])
    print(f"Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    from trl import SFTTrainer
    from transformers import TrainingArguments

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = eval_ds,
        dataset_text_field = "text",
        max_seq_length     = max_seq,
        args = TrainingArguments(
            output_dir                  = str(OUTPUT_DIR),
            num_train_epochs            = args.epochs,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps                = 50,
            learning_rate               = args.lr,
            fp16                        = False,
            bf16                        = True,   # Match Unsloth's bfloat16 model
            logging_steps               = 25,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            seed                        = 42,
            report_to                   = "none",
        ),
    )

    trainer.train()
    print("Training complete.")

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to GGUF → {GGUF_DIR}")
    model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

    gguf_files     = list(GGUF_DIR.glob("*.gguf"))
    gguf_path      = gguf_files[0] if gguf_files else GGUF_DIR / "model.gguf"
    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_path.write_text(
        f'FROM {gguf_path}\n'
        'PARAMETER temperature 0.7\n'
        'SYSTEM "You are a master storyteller and Dungeon Master for a D&D 5e campaign."\n'
    )

    print(f"\nollama create ragnarok-dm -f {modelfile_path}")
    print("ollama run ragnarok-dm")
    print("python orchestrator.py --local")


if __name__ == "__main__":
    main()

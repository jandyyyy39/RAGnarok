import argparse
import gc
import json
import itertools
from pathlib import Path

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE  = DATA_DIR / "fireball_train.jsonl"
EVAL_FILE   = DATA_DIR / "fireball_eval.jsonl"
RESULTS_FILE = DATA_DIR / "hparam_results.json"

TRAIN_SUBSET = 300
EVAL_SUBSET  = 75
MAX_SEQ_LEN  = 1024   # shorter context to keep each run fast
BATCH_SIZE   = 2
GRAD_ACCUM   = 4

DEFAULT_GRID = {
    "rank":          [8, 16, 32],
    "lora_alpha":    [8, 16, 32],
    "learning_rate": [5e-5, 1e-4, 2e-4],
}

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"


def load_jsonl(path: Path, limit: int = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


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


def run_single(rank: int, lora_alpha: int, learning_rate: float,
               train_ds, eval_ds, base_model, base_tokenizer) -> float:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    model = FastLanguageModel.get_peft_model(
        base_model,
        r                          = rank,
        lora_alpha                 = lora_alpha,
        lora_dropout               = 0.05,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = base_tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = eval_ds,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LEN,
        args = TrainingArguments(
            output_dir                  = str(DATA_DIR / "hparam_tmp"),
            num_train_epochs            = 1,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps                = 10,
            learning_rate               = learning_rate,
            fp16                        = True,
            logging_steps               = 50,
            eval_strategy               = "epoch",
            save_strategy               = "no",
            seed                        = 42,
            report_to                   = "none",
        ),
    )

    trainer.train()
    eval_results = trainer.evaluate()
    eval_loss    = round(eval_results.get("eval_loss", 99.0), 4)

    # Release LoRA adapter from GPU before next run
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return eval_loss


def print_table(results: list[dict]):
    print(f"\n{'Rank':<6} {'Alpha':<7} {'LR':<10} {'Eval Loss':<12} {'Status'}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["eval_loss"]):
        status = "<-- best" if r == sorted(results, key=lambda x: x["eval_loss"])[0] else ""
        print(f"{r['rank']:<6} {r['lora_alpha']:<7} {r['learning_rate']:<10} {r['eval_loss']:<12} {status}")


def main():
    parser = argparse.ArgumentParser(description="LoRA hyperparameter grid search on FIREBALL subset")
    parser.add_argument("--ranks",  nargs="+", type=int,   default=DEFAULT_GRID["rank"])
    parser.add_argument("--alphas", nargs="+", type=int,   default=DEFAULT_GRID["lora_alpha"])
    parser.add_argument("--lrs",    nargs="+", type=float, default=DEFAULT_GRID["learning_rate"])
    parser.add_argument("--train-samples", type=int, default=TRAIN_SUBSET)
    parser.add_argument("--eval-samples",  type=int, default=EVAL_SUBSET)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: run inside WSL2 with: pip install unsloth")
        return

    if not TRAIN_FILE.exists():
        print(f"ERROR: {TRAIN_FILE} not found. Run prepare_fireball.py first.")
        return

    combos = list(itertools.product(args.ranks, args.alphas, args.lrs))
    print(f"Grid search: {len(combos)} combinations")
    print(f"Subset: {args.train_samples} train / {args.eval_samples} eval samples, 1 epoch each\n")

    print(f"Loading base model {MODEL_ID}...")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_ID,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = False,
        dtype          = None,
    )

    train_raw = load_jsonl(TRAIN_FILE, limit=args.train_samples)
    eval_raw  = load_jsonl(EVAL_FILE,  limit=args.eval_samples)

    from datasets import Dataset
    train_ds = Dataset.from_list([{"text": format_prompt(s, base_tokenizer)} for s in train_raw])
    eval_ds  = Dataset.from_list([{"text": format_prompt(s, base_tokenizer)} for s in eval_raw])
    print(f"Subset loaded: {len(train_ds)} train / {len(eval_ds)} eval\n")

    results = []

    for i, (rank, alpha, lr) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] rank={rank}  alpha={alpha}  lr={lr}")
        try:
            eval_loss = run_single(rank, alpha, lr, train_ds, eval_ds, base_model, base_tokenizer)
            status = "ok"
        except Exception as e:
            print(f"  FAILED: {e}")
            eval_loss = 99.0
            status = "error"

        results.append({
            "rank": rank, "lora_alpha": alpha,
            "learning_rate": lr, "eval_loss": eval_loss, "status": status,
        })
        print(f"  eval_loss = {eval_loss}\n")

    best = min(results, key=lambda x: x["eval_loss"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"best": best, "all": results}, f, indent=2)

    print_table(results)

    print(f"\nBest config: rank={best['rank']}  alpha={best['lora_alpha']}  lr={best['learning_rate']}")
    print(f"Eval loss:   {best['eval_loss']}")
    print(f"\nRun full fine-tuning with:")
    print(f"  python scripts/finetune_fireball.py --rank {best['rank']} --alpha {best['lora_alpha']} --lr {best['learning_rate']}")
    print(f"\nFull results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()

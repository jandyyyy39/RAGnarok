"""
Generate training-process plots from all checkpoints under a LoRA directory.

Reads:
  <lora-dir>/checkpoint-*/trainer_state.json

Writes (default):
  data/plots/<name>/
    - <prefix>_train_loss.png
    - <prefix>_eval_loss.png
    - <prefix>_eval_perplexity.png
    - <prefix>_learning_rate.png
    - <prefix>_grad_norm.png
    - <prefix>_best_checkpoints.png

Example:
  python scripts/generate_lora_training_plots.py ^
    --lora-dir data/ragnarok-dm-lora ^
    --out-dir data/plots/ragnarok_training ^
    --prefix ragnarok_dm
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


def _checkpoint_step_from_name(name: str) -> int | None:
    # checkpoint-2847 -> 2847
    m = re.search(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def _load_trainer_state(fp: Path) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_curves(trainer_state: dict) -> dict:
    """
    Extract logged curves from HuggingFace Trainer `trainer_state.json`.
    """
    curves: dict[str, dict[int, float]] = {
        "loss": {},
        "eval_loss": {},
        "learning_rate": {},
        "grad_norm": {},
    }
    best_metric = trainer_state.get("best_metric")
    best_step = trainer_state.get("best_global_step")

    log_history = trainer_state.get("log_history", []) or []
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        try:
            step_i = int(step)
        except Exception:
            continue

        if "loss" in entry and entry.get("loss") is not None:
            curves["loss"][step_i] = float(entry["loss"])
        if "eval_loss" in entry and entry.get("eval_loss") is not None:
            curves["eval_loss"][step_i] = float(entry["eval_loss"])
        if "learning_rate" in entry and entry.get("learning_rate") is not None:
            curves["learning_rate"][step_i] = float(entry["learning_rate"])
        if "grad_norm" in entry and entry.get("grad_norm") is not None:
            curves["grad_norm"][step_i] = float(entry["grad_norm"])

    return {
        "curves": curves,
        "best_metric": float(best_metric) if best_metric is not None else None,
        "best_step": int(best_step) if best_step is not None else None,
    }


def _plot_series(
    x_steps: list[int],
    y_vals: list[float],
    out_path: Path,
    title: str,
    y_label: str,
    marker_step: int | None = None,
) -> None:
    plt.figure(figsize=(10.5, 6))
    plt.plot(x_steps, y_vals, linewidth=2.0, color="#1f77b4")
    if marker_step is not None and marker_step in x_steps:
        idx = x_steps.index(marker_step)
        plt.scatter([x_steps[idx]], [y_vals[idx]], color="#d62728", s=80, zorder=3)
        plt.annotate(
            f"best step\n{marker_step}",
            (x_steps[idx], y_vals[idx]),
            textcoords="offset points",
            xytext=(10, 15),
            ha="left",
            fontsize=10,
            color="#d62728",
        )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Training step", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LoRA training plots from checkpoint-*/trainer_state.json")
    p.add_argument("--lora-dir", type=str, required=True, help="Path to the LoRA directory (contains checkpoint-*/).")
    p.add_argument("--out-dir", type=str, default="", help="Output directory for PNGs (default: data/plots/<name>).")
    p.add_argument("--prefix", type=str, default="", help="Filename prefix for plots (default: directory name).")
    args = p.parse_args()

    lora_dir = Path(args.lora_dir).resolve()
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")

    name = lora_dir.name
    prefix = args.prefix.strip() or name
    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else lora_dir.parent / "plots" / name

    checkpoint_dirs = sorted([p for p in lora_dir.glob("checkpoint-*") if p.is_dir()])
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-*/ dirs found under: {lora_dir}")

    # Aggregate curves across checkpoints (dedup by step; last value wins).
    agg_curves: dict[str, dict[int, float]] = {
        "loss": {},
        "eval_loss": {},
        "learning_rate": {},
        "grad_norm": {},
    }
    ckpt_best: list[tuple[str, int | None, float | None]] = []

    for ckpt_dir in checkpoint_dirs:
        ts_fp = ckpt_dir / "trainer_state.json"
        if not ts_fp.is_file():
            continue
        ts = _load_trainer_state(ts_fp)
        extracted = _extract_curves(ts)

        for k in agg_curves.keys():
            for step_i, v in extracted["curves"][k].items():
                agg_curves[k][step_i] = v

        ckpt_step = _checkpoint_step_from_name(ckpt_dir.name)
        ckpt_best.append((ckpt_dir.name, extracted["best_step"] or ckpt_step, extracted["best_metric"]))

    # Pick best step (lowest eval_loss if present).
    best_step: int | None = None
    if agg_curves["eval_loss"]:
        best_step = min(agg_curves["eval_loss"].keys(), key=lambda s: agg_curves["eval_loss"][s])

    def sorted_steps_and_vals(key: str) -> tuple[list[int], list[float]]:
        items = sorted(agg_curves[key].items(), key=lambda kv: kv[0])
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        return xs, ys

    out_dir.mkdir(parents=True, exist_ok=True)

    if agg_curves["loss"]:
        xs, ys = sorted_steps_and_vals("loss")
        _plot_series(xs, ys, out_dir / f"{prefix}_train_loss.png", f"{prefix}: training loss", "Training loss", marker_step=best_step)

    if agg_curves["eval_loss"]:
        xs, ys = sorted_steps_and_vals("eval_loss")
        _plot_series(xs, ys, out_dir / f"{prefix}_eval_loss.png", f"{prefix}: eval loss", "Eval loss", marker_step=best_step)

        ppl_x = []
        ppl_y = []
        for x_i, v in zip(xs, ys):
            ppl_x.append(x_i)
            ppl_y.append(float(math.exp(v)))
        _plot_series(ppl_x, ppl_y, out_dir / f"{prefix}_eval_perplexity.png", f"{prefix}: eval perplexity (exp(eval_loss))", "Perplexity", marker_step=best_step)

    if agg_curves["learning_rate"]:
        xs, ys = sorted_steps_and_vals("learning_rate")
        _plot_series(xs, ys, out_dir / f"{prefix}_learning_rate.png", f"{prefix}: learning rate schedule", "Learning rate", marker_step=best_step)

    if agg_curves["grad_norm"]:
        xs, ys = sorted_steps_and_vals("grad_norm")
        _plot_series(xs, ys, out_dir / f"{prefix}_grad_norm.png", f"{prefix}: grad norm (stability)", "Grad norm", marker_step=best_step)

    # Best checkpoint metadata plot (bar chart).
    if ckpt_best:
        filtered = [(ckpt, step, metric) for ckpt, step, metric in ckpt_best if step is not None or metric is not None]
        filtered = sorted(filtered, key=lambda t: (t[1] if t[1] is not None else 10**18))
        if filtered:
            labels = [ck for ck, _, _ in filtered]
            x = list(range(len(filtered)))
            y = [m if m is not None else float("nan") for _, _, m in filtered]
            plt.figure(figsize=(10.5, 6))
            plt.bar(x, y, color="#2ca02c")
            plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
            plt.title(f"{prefix}: best_metric per checkpoint", fontsize=13, fontweight="bold")
            plt.ylabel("best_metric", fontsize=12)
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_best_checkpoints.png", dpi=200)
            plt.close()

    print(f"Generated LoRA training plots -> {out_dir}")
    for fp in sorted(out_dir.glob(f"{prefix}_*.png")):
        print(" -", fp.name)


if __name__ == "__main__":
    main()


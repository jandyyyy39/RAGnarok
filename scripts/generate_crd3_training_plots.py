"""
Generate CRD3 fine-tuning training plots from all LoRA checkpoints.

Reads:
  data/crd3-llama32-r16a32-lr2e4-lora/checkpoint-*/trainer_state.json

Writes:
  data/plots/crd3_training/
    - crd3_train_loss.png
    - crd3_eval_loss.png
    - crd3_eval_perplexity.png
    - crd3_learning_rate.png
    - crd3_grad_norm.png
    - crd3_best_checkpoints.png

Run:
  python scripts/generate_crd3_training_plots.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
LORA_DIR = ROOT / "data" / "crd3-llama32-r16a32-lr2e4-lora"
PLOTS_DIR = ROOT / "data" / "plots" / "crd3_training"


def _checkpoint_step_from_name(name: str) -> int | None:
    # checkpoint-2847 -> 2847
    m = re.search(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def _load_trainer_state(fp: Path) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_curves(trainer_state: dict) -> dict:
    """
    Extract time series from HuggingFace Trainer `trainer_state.json`.

    We keep the most recently seen value for each `step` if duplicates occur.
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
    y_formatter=None,
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
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    if not LORA_DIR.exists():
        raise FileNotFoundError(f"LoRA directory not found: {LORA_DIR}")

    checkpoint_dirs = sorted([p for p in LORA_DIR.glob("checkpoint-*") if p.is_dir()])
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-*/ dirs found under {LORA_DIR}")

    # Aggregate curves across checkpoints (dedup by step).
    agg_curves: dict[str, dict[int, float]] = {
        "loss": {},
        "eval_loss": {},
        "learning_rate": {},
        "grad_norm": {},
    }

    best_steps: list[tuple[str, int, float | None]] = []
    for ckpt_dir in checkpoint_dirs:
        trainer_state_fp = ckpt_dir / "trainer_state.json"
        if not trainer_state_fp.is_file():
            continue
        trainer_state = _load_trainer_state(trainer_state_fp)
        extracted = _extract_curves(trainer_state)

        # Merge time series: last one wins for each step.
        for k in agg_curves.keys():
            for step_i, v in extracted["curves"][k].items():
                agg_curves[k][step_i] = v

        ckpt_step = _checkpoint_step_from_name(ckpt_dir.name) or -1
        best_steps.append(
            (
                ckpt_dir.name,
                extracted["best_step"] if extracted["best_step"] is not None else ckpt_step,
                extracted["best_metric"],
            )
        )

    # Determine which step is best across checkpoints (lowest eval_loss if available, else best_metric).
    best_step: int | None = None
    if agg_curves["eval_loss"]:
        # lowest eval_loss value wins
        best_step = min(agg_curves["eval_loss"].keys(), key=lambda s: agg_curves["eval_loss"][s])
    else:
        # fallback: best_metric across trainer_state values
        scored = [(s, m) for _, s, m in best_steps if m is not None]
        if scored:
            # best_metric in this run is eval_loss-like, so lower is better
            best_step = min(scored, key=lambda t: t[1])[0]

    # Prepare series for plotting.
    def sorted_steps_and_vals(key: str) -> tuple[list[int], list[float]]:
        items = sorted(agg_curves[key].items(), key=lambda kv: kv[0])
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        return xs, ys

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if agg_curves["loss"]:
        xs, ys = sorted_steps_and_vals("loss")
        _plot_series(
            xs,
            ys,
            PLOTS_DIR / "crd3_train_loss.png",
            title="CRD3 LoRA training loss",
            y_label="Training loss",
            marker_step=best_step,
        )

    if agg_curves["eval_loss"]:
        xs, ys = sorted_steps_and_vals("eval_loss")
        _plot_series(
            xs,
            ys,
            PLOTS_DIR / "crd3_eval_loss.png",
            title="CRD3 LoRA eval loss",
            y_label="Eval loss",
            marker_step=best_step,
        )
        # Perplexity = exp(loss)
        ppl = [math.exp(v) if v is not None else None for v in ys]
        # Filter None (shouldn't happen)
        xs2: list[int] = []
        ppl2: list[float] = []
        for x_i, p in zip(xs, ppl):
            if p is None:
                continue
            xs2.append(x_i)
            ppl2.append(float(p))
        _plot_series(
            xs2,
            ppl2,
            PLOTS_DIR / "crd3_eval_perplexity.png",
            title="CRD3 LoRA eval perplexity (exp(eval_loss))",
            y_label="Perplexity",
            marker_step=best_step,
        )

    if agg_curves["learning_rate"]:
        xs, ys = sorted_steps_and_vals("learning_rate")
        _plot_series(
            xs,
            ys,
            PLOTS_DIR / "crd3_learning_rate.png",
            title="CRD3 LoRA learning rate schedule",
            y_label="Learning rate",
            marker_step=best_step,
        )

    if agg_curves["grad_norm"]:
        xs, ys = sorted_steps_and_vals("grad_norm")
        _plot_series(
            xs,
            ys,
            PLOTS_DIR / "crd3_grad_norm.png",
            title="CRD3 LoRA gradient norm (stability)",
            y_label="Grad norm",
            marker_step=best_step,
        )

    # Best checkpoint summary: show best step/metric per checkpoint file (for transparency).
    if best_steps:
        # pick rows that actually contain best info
        filtered = [(ckpt, step, metric) for ckpt, step, metric in best_steps if step is not None]
        filtered = sorted(filtered, key=lambda t: t[1])
        plt.figure(figsize=(10.5, 6))
        x_labels = [ck for ck, _, _ in filtered]
        x = list(range(len(filtered)))
        y = [m if m is not None else float("nan") for _, _, m in filtered]
        plt.bar(x, y, color="#2ca02c")
        plt.xticks(x, x_labels, rotation=45, ha="right", fontsize=8)
        plt.title("CRD3 LoRA checkpoint metadata (best_metric per trainer_state)", fontsize=13, fontweight="bold")
        plt.ylabel("best_metric (lower=better if it's eval_loss)", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "crd3_best_checkpoints.png", dpi=200)
        plt.close()

    print(f"Generated training plots -> {PLOTS_DIR}")
    print("Files:")
    for fp in sorted(PLOTS_DIR.glob("crd3_*.png")):
        print(" -", fp.name)


if __name__ == "__main__":
    main()


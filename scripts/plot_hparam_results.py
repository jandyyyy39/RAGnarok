"""
Visualize data/hparam_results.json for presentations.

Perplexity is computed as exp(eval_loss). HuggingFace Trainer reports mean token
cross-entropy in nats; perplexity = exp(NLL) is standard and preserves ranking.

Outputs (loss and perplexity variants):
  hparam_heatmaps.png / hparam_heatmaps_perplexity.png
  hparam_ranked_bars.png / hparam_ranked_bars_perplexity.png
  hparam_grid_all.png / hparam_grid_all_perplexity.png

Usage:
  python scripts/plot_hparam_results.py
  python scripts/plot_hparam_results.py --metric loss-only
  python scripts/plot_hparam_results.py --metric perplexity-only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

COLORS = {"best": "#2563eb", "other": "#cbd5e1", "line": "#0f766e"}


def load_results(path: Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["best"], data["all"]


def _prepare_metric_column(df: pd.DataFrame, use_perplexity: bool) -> pd.DataFrame:
    """Add _plot column: eval_loss or exp(eval_loss) for perplexity."""
    d = df.copy()
    if use_perplexity:
        d["_plot"] = np.exp(d["eval_loss"].astype(float))
    else:
        d["_plot"] = d["eval_loss"].astype(float)
    return d


def _best_metric_value(best: dict, use_perplexity: bool) -> float:
    if use_perplexity:
        if best.get("eval_perplexity") is not None:
            return float(best["eval_perplexity"])
        return float(math.exp(float(best["eval_loss"])))
    return float(best["eval_loss"])


def plot_heatmaps(
    df: pd.DataFrame, best: dict, out_dir: Path, *, use_perplexity: bool, file_tag: str
) -> None:
    df = _prepare_metric_column(df, use_perplexity)
    lrs = sorted(df["learning_rate"].unique())
    n = len(lrs)
    vmin, vmax = float(df["_plot"].min()), float(df["_plot"].max())
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for ax, lr in zip(axes[0], lrs):
        sub = df[df["learning_rate"] == lr].pivot_table(
            index="rank", columns="lora_alpha", values="_plot", aggfunc="first"
        )
        ranks = sub.index.values
        alphas = sub.columns.values
        mat = sub.values.astype(float)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([str(int(a)) for a in alphas])
        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels([str(int(r)) for r in ranks])
        fmt = "{:.2f}" if use_perplexity else "{:.3f}"
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(j, i, fmt.format(val), ha="center", va="center", color="black", fontsize=11)
        ax.set_xlabel("LoRA α")
        ax.set_ylabel("Rank (r)")
        ax.set_title(f"Learning rate = {lr:g}")

    label = "Perplexity (lower is better)" if use_perplexity else "Eval loss (cross-entropy, lower is better)"
    fig.colorbar(im, ax=axes[0].ravel().tolist(), shrink=0.6, label=label)
    title = "Hyperparameter search — " + ("perplexity" if use_perplexity else "eval loss")
    fig.suptitle(title + " (lower is better)", fontsize=15, y=1.02)
    out = out_dir / f"hparam_heatmaps{file_tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_ranked_bars(df: pd.DataFrame, best: dict, out_dir: Path, *, use_perplexity: bool, file_tag: str) -> None:
    df = _prepare_metric_column(df, use_perplexity)
    d = df.sort_values("eval_loss").reset_index(drop=True)
    labels = [
        f"r={int(r)}  α={int(a)}  lr={lr:g}"
        for r, a, lr in zip(d["rank"], d["lora_alpha"], d["learning_rate"])
    ]

    def is_best(r, a, lr) -> bool:
        return (
            int(r) == int(best["rank"])
            and int(a) == int(best["lora_alpha"])
            and abs(float(lr) - float(best["learning_rate"])) < 1e-9
        )

    colors = [
        COLORS["best"] if is_best(r, a, lr) else COLORS["other"]
        for r, a, lr in zip(d["rank"], d["lora_alpha"], d["learning_rate"])
    ]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.32 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["_plot"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    if use_perplexity:
        ax.set_xlabel("Perplexity (exp of cross-entropy eval loss)")
        ax.set_title("All LoRA configs ranked — lowest perplexity = same as lowest loss")
    else:
        ax.set_xlabel("Eval loss (cross-entropy on held-out subset)")
        ax.set_title("All LoRA configs ranked — lowest loss = selected for full fine-tune")
    bv = _best_metric_value(best, use_perplexity)
    ax.axvline(
        bv,
        color=COLORS["line"],
        linestyle="--",
        alpha=0.9,
        linewidth=1.5,
        label=("Best perplexity = " if use_perplexity else "Best loss = ")
        + (f"{bv:.2f}" if use_perplexity else f"{bv:.4f}"),
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = out_dir / f"hparam_ranked_bars{file_tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_full_hyperparameter_grid(
    df: pd.DataFrame, best: dict, out_dir: Path, *, use_perplexity: bool, file_tag: str
) -> None:
    df = _prepare_metric_column(df, use_perplexity)
    ranks = sorted(df["rank"].unique())
    lrs = sorted(df["learning_rate"].unique())
    alphas = sorted(df["lora_alpha"].unique())

    n_r, n_c = len(ranks), len(lrs) * len(alphas)
    mat = np.full((n_r, n_c), np.nan)
    col_meta: list[tuple[float, int]] = []
    for lr in lrs:
        for a in alphas:
            col_meta.append((lr, a))

    best_ij: tuple[int, int] | None = None
    for j, (lr, a) in enumerate(col_meta):
        for i, r in enumerate(ranks):
            row = df[(df["rank"] == r) & (df["learning_rate"] == lr) & (df["lora_alpha"] == a)]
            if len(row) != 1:
                continue
            v = float(row["_plot"].iloc[0])
            mat[i, j] = v
            if (
                int(r) == int(best["rank"])
                and int(a) == int(best["lora_alpha"])
                and abs(float(lr) - float(best["learning_rate"])) < 1e-9
            ):
                best_ij = (i, j)

    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    fig_w = max(14, 1.2 * n_c)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r", vmin=vmin, vmax=vmax)

    col_labels = [f"lr={lr:g}\nα={int(a)}" for lr, a in col_meta]
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels([f"r = {int(r)}" for r in ranks])
    ax.set_xlabel("Learning rate × LoRA α (each column is one combination)")
    ax.set_ylabel("Rank (r)")

    fmt = "{:.2f}" if use_perplexity else "{:.3f}"
    for i in range(n_r):
        for j in range(n_c):
            val = mat[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, fmt.format(val), ha="center", va="center", color="black", fontsize=10, fontweight="medium")

    if best_ij is not None:
        bi, bj = best_ij
        ax.add_patch(
            Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="#059669", linewidth=3)
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Perplexity (lower is better)" if use_perplexity else "Eval loss (lower is better)")
    ax.set_title(
        "Hyperparameter grid — all combinations (green = selected)"
        + (" (perplexity)" if use_perplexity else " (cross-entropy)")
    )
    plt.tight_layout()
    out = out_dir / f"hparam_grid_all{file_tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=Path("data/hparam_results.json"))
    p.add_argument("--out", type=Path, default=Path("data/plots"))
    p.add_argument(
        "--metric",
        choices=("both", "loss-only", "perplexity-only"),
        default="both",
        help="Which figures to generate (default: loss + perplexity)",
    )
    args = p.parse_args()

    if not args.json.is_file():
        print(f"ERROR: {args.json} not found")
        return

    best, rows = load_results(args.json)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)

    bp = best.get("eval_perplexity")
    if bp is None:
        bp = math.exp(float(best["eval_loss"]))
    print(
        f"Best: rank={best['rank']} alpha={best['lora_alpha']} lr={best['learning_rate']} "
        f"loss={best['eval_loss']} perplexity~{bp:.4f}"
    )

    do_loss = args.metric in ("both", "loss-only")
    do_ppl = args.metric in ("both", "perplexity-only")

    if do_loss:
        plot_heatmaps(df, best, args.out, use_perplexity=False, file_tag="")
        plot_ranked_bars(df, best, args.out, use_perplexity=False, file_tag="")
        plot_full_hyperparameter_grid(df, best, args.out, use_perplexity=False, file_tag="")
    if do_ppl:
        plot_heatmaps(df, best, args.out, use_perplexity=True, file_tag="_perplexity")
        plot_ranked_bars(df, best, args.out, use_perplexity=True, file_tag="_perplexity")
        plot_full_hyperparameter_grid(df, best, args.out, use_perplexity=True, file_tag="_perplexity")


if __name__ == "__main__":
    main()

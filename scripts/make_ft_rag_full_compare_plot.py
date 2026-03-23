"""
Make a single clustered bar chart comparing:
  - FT+RAG (CRD3) vs Full model (CRD3)
  - FT+RAG (Fireball) vs Full model (Fireball)

It reads the existing ablation evaluation JSON outputs:
  CRD3:     data/eval_crd3_ablation_{full_pipeline_groq,local_ft_rag_on}.json
  FIREBALL: data/eval_results8_{full_pipeline_groq,local_ft_rag_on}.json

Metrics: ROUGE-L, BERTScore, BLEU, chrF, Rule Coverage, distinct-1, LLM Composite (normalized to 0–1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


def _load_json(fp: Path) -> list[dict]:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric_means(results: list[dict], metrics: list[str]) -> dict[str, float]:
    agg: dict[str, list[float]] = {m: [] for m in metrics}
    for r in results:
        scores = r.get("scores", {}) or {}
        for m in metrics:
            v = scores.get(m)
            if v is None:
                continue
            agg[m].append(v)
    return {m: _safe_mean(vs) for m, vs in agg.items()}


def _normalize_metric(m: str, v: float) -> float:
    # Keep consistent with evaluate_ablation_combine.py:
    # llm_composite is stored as 1–5 (avg of rubric), normalize to 0–1.
    if m == "llm_composite":
        return v / 5.0
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/plots/ft_rag_full_compare_grouped.png")
    args = p.parse_args()

    # Files
    crd3_full_fp = DATA / "eval_crd3_ablation_full_pipeline_groq.json"
    crd3_ft_fp = DATA / "eval_crd3_ablation_local_ft_rag_on.json"
    fire_full_fp = DATA / "eval_results8_full_pipeline_groq.json"
    fire_ft_fp = DATA / "eval_results8_local_ft_rag_on.json"

    for fp in [crd3_full_fp, crd3_ft_fp, fire_full_fp, fire_ft_fp]:
        if not fp.is_file():
            raise FileNotFoundError(f"Missing: {fp}")

    metrics = ["rougeL", "bertscore", "bleu", "chrf", "rule_coverage", "distinct1", "llm_composite"]

    labels = [
        "FT+RAG (CRD3)",
        "Full model (CRD3)",
        "FT+RAG (Fireball)",
        "Full model (Fireball)",
    ]
    fps = [crd3_ft_fp, crd3_full_fp, fire_ft_fp, fire_full_fp]

    series: list[dict[str, float]] = []
    for fp in fps:
        res = _load_json(fp)
        series.append(_extract_metric_means(res, metrics))

    # Normalize for llm_composite
    series_norm = []
    for d in series:
        series_norm.append({m: _normalize_metric(m, d[m]) for m in metrics})

    # Clustered bar chart
    x = np.arange(len(labels))
    width = 0.12
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#DA8BC3", "#2CA02C"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(metrics):
        offset = (i - len(metrics) / 2 + 0.5) * width
        vals = [d[m] for d in series_norm]
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
            label=m.replace("_", " ").title(),
        )
        # Add numeric labels for each bar to make the chart self-contained.
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score (normalized 0–1)")
    ax.set_title("FT+RAG vs Full model (CRD3 vs Fireball)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    out_fp = (ROOT / args.out).resolve() if not args.out.startswith("/") else Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fp, dpi=200)
    plt.close()
    print(f"Saved -> {out_fp}")


if __name__ == "__main__":
    main()


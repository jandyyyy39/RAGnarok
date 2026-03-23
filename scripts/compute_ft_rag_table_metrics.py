"""
Compute table metrics used for comparing:
  - Vanilla (full pipeline, Groq) on FIREBALL eval_results8
  - FIREBALL FT (Local FT + RAG On) on eval_results8
  - CRD3 FT (Local FT + RAG On) on eval_crd3_ablation

Quality Composite definition (matches project scatter):
  mean(rougeL, bertscore, chrf)  (all are already 0–1)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def load(fp: Path):
    return json.loads(fp.read_text(encoding="utf-8"))


def quality_composite(scores: dict) -> float:
    return safe_mean([scores.get("rougeL"), scores.get("bertscore"), scores.get("chrf")])


def summarize(fp: Path, label: str):
    res = load(fp)
    lats = [r.get("latency_ms") for r in res if r.get("latency_ms") is not None]
    comps = [quality_composite((r.get("scores") or {})) for r in res]
    mean_lat = safe_mean(lats)
    mean_comp = safe_mean(comps)
    narrative = safe_mean([(r.get("scores") or {}).get("narrative_quality") for r in res])
    rules_acc = safe_mean([(r.get("scores") or {}).get("rules_accuracy") for r in res])
    voice = safe_mean([(r.get("scores") or {}).get("character_voice") for r in res])
    # Also print means of each component.
    components = ["rougeL", "bertscore", "chrf"]
    means = {}
    for c in components:
        vals = [(r.get("scores") or {}).get(c) for r in res]
        vals = [v for v in vals if v is not None]
        means[c] = safe_mean(vals)
    print(label)
    print("  n_inputs:", len(res))
    print("  avg_latency_ms:", round(mean_lat, 3))
    print("  quality_composite_mean:", round(mean_comp, 6))
    print("  judge_means (1–5):", {
        "narrative_quality": round(narrative, 3),
        "rules_accuracy": round(rules_acc, 3),
        "character_voice": round(voice, 3),
    })
    print("  mean_components:", {k: round(v, 6) for k, v in means.items()})


def main():
    summarize(DATA / "eval_results8_full_pipeline_groq.json", "Vanilla (Fireball, Full Pipeline Groq)")
    summarize(DATA / "eval_results8_local_ft_rag_on.json", "FT+RAG (Fireball, Local FT RAG On)")
    summarize(DATA / "eval_crd3_ablation_local_ft_rag_on.json", "FT+RAG (CRD3, Local FT RAG On)")
    summarize(DATA / "eval_crd3_ablation_full_pipeline_groq.json", "Full model (CRD3, Full Pipeline Groq)")


if __name__ == "__main__":
    main()


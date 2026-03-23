"""
Compute baseline (local mistral) quality composite + latency means.
Quality Composite = mean(rougeL, bertscore, chrf) per sample.
Then take mean across samples.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "data" / "eval_results_local_mistral.json"


def safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    res = json.loads(FP.read_text(encoding="utf-8"))
    comps = []
    lats = []
    for r in res:
        lats.append(r.get("latency_ms"))
        s = r.get("scores", {}) or {}
        comps.append(safe_mean([s.get("rougeL"), s.get("bertscore"), s.get("chrf")]))

    print("n_inputs:", len(res))
    print("avg_latency_ms:", round(safe_mean(lats), 3))
    print("quality_composite_mean:", round(safe_mean(comps), 6))

    # Judge means if present
    for k in ["narrative_quality", "rules_accuracy", "character_voice"]:
        vals = [((r.get("scores", {}) or {}).get(k)) for r in res]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"{k}_mean:", round(safe_mean(vals), 3))


if __name__ == "__main__":
    main()


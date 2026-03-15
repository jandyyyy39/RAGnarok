import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config

DATA_DIR     = ROOT / "data"
RESULTS_FILE = DATA_DIR / "eval_results.json"
PLOTS_DIR    = DATA_DIR / "plots"
FIREBALL_EVAL = Config.EVAL_FILE

EXPERIMENTS = [
    {"name": "Full Pipeline (Groq)",  "use_local": False, "use_rag": True,  "use_memory": True,  "use_npc": True},
    {"name": "No RAG",                "use_local": False, "use_rag": False, "use_memory": True,  "use_npc": True},
    {"name": "No Memory",             "use_local": False, "use_rag": True,  "use_memory": False, "use_npc": True},
    {"name": "No NPC Pass",           "use_local": False, "use_rag": True,  "use_memory": True,  "use_npc": False},
    {"name": "Bare DM Only",          "use_local": False, "use_rag": False, "use_memory": False, "use_npc": False},
    {"name": "Local (Mistral)",       "use_local": True,  "use_rag": True,  "use_memory": True,  "use_npc": True},
    {"name": "Local FT + No RAG",     "use_local": True,  "use_rag": False, "use_memory": True,  "use_npc": True},
]

# D&D mechanical terms — checks whether the pipeline's ruling is reflected in output
RULE_KEYWORDS = [
    "roll", "check", "saving throw", "dc", "attack", "damage", "hit points",
    "ability", "modifier", "advantage", "disadvantage", "proficiency",
    "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma",
    "action", "bonus action", "reaction", "spell", "armor class",
]

METRICS = ["rouge1", "rouge2", "rougeL", "bertscore", "rule_coverage", "response_length"]


# ---------------------------------------------------------------------------
# Metric Functions — fully local, no API
# ---------------------------------------------------------------------------

def compute_rouge(hypothesis: str, reference: str) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        print("  [ROUGE] rouge-score not installed: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore(hypothesis: str, reference: str) -> float:
    try:
        from bert_score import score as bert_score
        import warnings
        warnings.filterwarnings("ignore")
        _, _, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
        return round(F1[0].item(), 4)
    except ImportError:
        print("  [BERTScore] bert-score not installed: pip install bert-score")
        return 0.0


def compute_rule_coverage(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in RULE_KEYWORDS if kw in text_lower)
    return round(hits / len(RULE_KEYWORDS), 4)


def score_output(hypothesis: str, reference: str) -> dict:
    rouge  = compute_rouge(hypothesis, reference)
    bscore = compute_bertscore(hypothesis, reference)
    rule   = compute_rule_coverage(hypothesis)
    length = len(hypothesis.split())

    return {
        "rouge1":          rouge["rouge1"],
        "rouge2":          rouge["rouge2"],
        "rougeL":          rouge["rougeL"],
        "bertscore":       bscore,
        "rule_coverage":   rule,
        "response_length": length,
    }


# ---------------------------------------------------------------------------
# Load test set from FIREBALL eval split (ground-truth DM narrations)
# ---------------------------------------------------------------------------

def load_test_inputs(limit: int = 25) -> list[dict]:
    if not FIREBALL_EVAL.exists():
        print(f"WARNING: {FIREBALL_EVAL} not found.")
        print("Run scripts/prepare_fireball.py first, or falling back to built-in inputs.")
        return FALLBACK_INPUTS[:limit]

    samples = []
    with open(FIREBALL_EVAL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                samples.append({
                    "input":     row["input"],
                    "reference": row["output"],
                    "type":      "fireball",
                })
            if len(samples) >= limit:
                break

    print(f"Loaded {len(samples)} test inputs from FIREBALL eval set.")
    return samples


# Fallback if FIREBALL data not yet prepared
FALLBACK_INPUTS = [
    {"input": "I draw my sword and attack the goblin chief.",                     "reference": "", "type": "combat"},
    {"input": "I cast Fireball at the cluster of skeletons.",                     "reference": "", "type": "combat"},
    {"input": "I try to grapple the orc and pin him against the wall.",           "reference": "", "type": "combat"},
    {"input": "I tell Durnan I'm looking for info about the Undermountain.",      "reference": "", "type": "roleplay"},
    {"input": "I try to persuade the guard to let us through without paying.",    "reference": "", "type": "roleplay"},
    {"input": "I search the dead guard's body for keys or valuables.",            "reference": "", "type": "exploration"},
    {"input": "I carefully examine the runes carved into the dungeon door.",      "reference": "", "type": "exploration"},
    {"input": "Can I cast two spells in the same turn?",                          "reference": "", "type": "rules"},
    {"input": "I use my Bonus Action to hide after attacking.",                   "reference": "", "type": "rules"},
    {"input": "Natural 20 on my attack roll!",                                    "reference": "", "type": "dice"},
    {"input": "I drink the mysterious glowing potion I found in the chest.",      "reference": "", "type": "complex"},
    {"input": "I try to reason with the dragon instead of fighting it.",          "reference": "", "type": "complex"},
]


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(test_inputs: list[dict]) -> list[dict]:
    from orchestrator import RAGnarokOrchestrator

    all_results = []

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"CONDITION: {exp['name']}")
        print(f"{'='*60}")

        if exp["use_local"]:
            try:
                import requests
                requests.get("http://localhost:11434", timeout=2)
            except Exception:
                print("  [SKIP] Ollama not running. Start with: ollama serve")
                continue

        try:
            system = RAGnarokOrchestrator(
                use_local     = exp["use_local"],
                use_rag       = exp["use_rag"],
                use_memory    = exp["use_memory"],
                use_npc       = exp["use_npc"],
                use_voice     = False,
                smart_routing = False,
            )
        except Exception as e:
            print(f"  [SKIP] Init failed: {e}")
            continue

        for i, test in enumerate(test_inputs):
            player_input = test["input"]
            reference    = test.get("reference", "")
            print(f"  [{i+1}/{len(test_inputs)}] {player_input[:60]}...")

            turn       = system.process_turn(player_input)
            hypothesis = turn.get("response", "")

            if reference:
                scores = score_output(hypothesis, reference)
            else:
                scores = {
                    "rouge1":          None,
                    "rouge2":          None,
                    "rougeL":          None,
                    "bertscore":       None,
                    "rule_coverage":   compute_rule_coverage(hypothesis),
                    "response_length": len(hypothesis.split()),
                }

            result = {
                "experiment":   exp["name"],
                "input":        player_input,
                "input_type":   test.get("type", "unknown"),
                "response":     hypothesis,
                "reference":    reference,
                "latency_ms":   turn.get("latency_ms", 0),
                "skipped":      turn.get("skipped", []),
                "scores":       scores,
            }
            all_results.append(result)
            print(f"    ROUGE-L={scores.get('rougeL')}  BERTScore={scores.get('bertscore')}  "
                  f"RuleCov={scores['rule_coverage']}  {turn.get('latency_ms')}ms")

    return all_results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        exp = r["experiment"]
        agg[exp]["latency"].append(r["latency_ms"])
        for k, v in r["scores"].items():
            if v is not None:
                agg[exp][k].append(v)

    print("\n" + "=" * 90)
    print(f"{'Configuration':<28} {'ROUGE-L':>8} {'BERTScore':>10} {'RuleCov':>9} {'Length':>8} {'Latency':>10}")
    print("-" * 90)
    for exp, metrics in agg.items():
        def avg(k): return round(sum(metrics[k]) / len(metrics[k]), 3) if metrics[k] else "-"
        lat = round(sum(metrics["latency"]) / len(metrics["latency"])) if metrics["latency"] else 0
        print(f"{exp:<28} {str(avg('rougeL')):>8} {str(avg('bertscore')):>10} "
              f"{str(avg('rule_coverage')):>9} {str(avg('response_length')):>8} {lat:>9}ms")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def build_plots(results: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Plots] pip install matplotlib")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        exp = r["experiment"]
        agg[exp]["latency"].append(r["latency_ms"])
        for k, v in r["scores"].items():
            if v is not None:
                agg[exp][k].append(v)

    experiments   = list(agg.keys())
    short_names   = [e.replace("Full Pipeline ", "Full\n").replace("Local FT", "FT") for e in experiments]
    COLORS        = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]

    def avg(exp, k): return round(sum(agg[exp][k]) / len(agg[exp][k]), 3) if agg[exp][k] else 0

    plot_metrics  = ["rouge1", "rouge2", "rougeL", "bertscore", "rule_coverage"]
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "Rule Coverage"]
    avg_latency   = [round(sum(agg[e]["latency"]) / max(len(agg[e]["latency"]), 1)) for e in experiments]

    # ── Plot 1: Grouped metric bars ──────────────────────────────────────────
    n_exp, n_met = len(experiments), len(plot_metrics)
    x     = np.arange(n_exp)
    width = 0.15
    met_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (m, color, label) in enumerate(zip(plot_metrics, met_colors, metric_labels)):
        vals   = [avg(e, m) for e in experiments]
        offset = (i - n_met / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_title("Per-Metric Scores by Configuration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "scores_grouped.png", dpi=150); plt.close()
    print(f"  Saved → scores_grouped.png")

    # ── Plot 2: ROUGE-L only (primary quality metric) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [avg(e, "rougeL") for e in experiments]
    bars = ax.bar(short_names, vals, color=COLORS[:n_exp], width=0.6, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("ROUGE-L F1", fontsize=11)
    ax.set_title("ROUGE-L Score by Configuration", fontsize=13, fontweight="bold")
    ax.axhline(vals[0], color="gray", linestyle="--", linewidth=0.8, label="Baseline")
    ax.legend(fontsize=9); plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "scores_rougeL.png", dpi=150); plt.close()
    print(f"  Saved → scores_rougeL.png")

    # ── Plot 3: Latency ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_names, avg_latency, color=COLORS[:n_exp], width=0.6, edgecolor="white")
    ax.bar_label(bars, fmt="%dms", padding=3, fontsize=9)
    ax.set_ylabel("Average Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Latency by Configuration", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "latency.png", dpi=150); plt.close()
    print(f"  Saved → latency.png")

    # ── Plot 4: ROUGE-L vs Latency scatter ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (exp, color) in enumerate(zip(experiments, COLORS)):
        ax.scatter(avg_latency[i], avg(exp, "rougeL"), s=120, color=color,
                   label=short_names[i], zorder=3)
        ax.annotate(short_names[i], (avg_latency[i], avg(exp, "rougeL")),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Average Latency (ms)", fontsize=11)
    ax.set_ylabel("ROUGE-L F1", fontsize=11)
    ax.set_title("Quality vs. Latency Trade-off", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "quality_vs_latency.png", dpi=150); plt.close()
    print(f"  Saved → quality_vs_latency.png")

    # ── Plot 5: Heatmap (config × metric) ────────────────────────────────────
    matrix = np.array([[avg(e, m) for m in plot_metrics] for e in experiments])
    fig, ax = plt.subplots(figsize=(9, max(4, n_exp * 0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_met)); ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(n_exp)); ax.set_yticklabels(short_names, fontsize=9)
    for i in range(n_exp):
        for j in range(n_met):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Score (0–1)")
    ax.set_title("Score Heatmap: Configuration × Metric", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "heatmap.png", dpi=150); plt.close()
    print(f"  Saved → heatmap.png")

    # ── Plot 6: Radar chart (top 4 configs by ROUGE-L) ───────────────────────
    top4 = sorted(experiments, key=lambda e: avg(e, "rougeL"), reverse=True)[:4]
    N     = len(plot_metrics)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    radar_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for exp, color in zip(top4, radar_colors):
        vals = [avg(exp, m) for m in plot_metrics] + [avg(exp, plot_metrics[0])]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=exp)
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1); ax.set_title("Radar: Top 4 Configurations", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "radar.png", dpi=150); plt.close()
    print(f"  Saved → radar.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGnarok Evaluation Harness")
    parser.add_argument("--plot-only", action="store_true", help="Re-generate plots from existing results")
    parser.add_argument("--quick",     action="store_true", help="Use only 8 test inputs")
    parser.add_argument("--n",         type=int, default=25, help="Number of test inputs to use")
    args = parser.parse_args()

    if args.plot_only:
        if not RESULTS_FILE.exists():
            print(f"ERROR: {RESULTS_FILE} not found. Run without --plot-only first.")
            sys.exit(1)
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        limit   = 8 if args.quick else args.n
        inputs  = load_test_inputs(limit=limit)
        print(f"\nEvaluating {len(EXPERIMENTS)} configurations × {len(inputs)} inputs")
        print("Metrics: ROUGE-1/2/L, BERTScore, Rule Coverage, Latency (no API required)\n")
        results = run_evaluation(inputs)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {RESULTS_FILE}")

    print_summary(results)
    print("\nGenerating plots...")
    build_plots(results)

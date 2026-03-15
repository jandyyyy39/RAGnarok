"""
evaluate.py
-----------
Runs all experimental configurations (flag combinations) through a curated
test set and scores each output with an LLM-as-judge rubric.

Produces:
  data/eval_results.json        — raw scores + metadata for every run
  data/plots/scores_grouped.png — grouped bar chart: metric scores per config
  data/plots/scores_avg.png     — average composite score per config
  data/plots/latency.png        — end-to-end latency per config
  data/plots/radar.png          — radar chart comparing top configurations
  data/plots/heatmap.png        — score heatmap (config × metric)

Usage:
    # Full run (calls LLM for every test input × config)
    python scripts/evaluate.py

    # Re-generate plots from a previous run (no LLM calls)
    python scripts/evaluate.py --plot-only

    # Run a quick smoke-test with fewer inputs
    python scripts/evaluate.py --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groq import Groq
from config import Config

DATA_DIR    = ROOT / "data"
RESULTS_FILE = DATA_DIR / "eval_results.json"
PLOTS_DIR   = DATA_DIR / "plots"

# ---------------------------------------------------------------------------
# Experimental Conditions
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "name":       "Full Pipeline (Groq)",
        "use_local":  False, "use_rag": True,
        "use_memory": True,  "use_npc": True,
    },
    {
        "name":       "No RAG",
        "use_local":  False, "use_rag": False,
        "use_memory": True,  "use_npc": True,
    },
    {
        "name":       "No Memory",
        "use_local":  False, "use_rag": True,
        "use_memory": False, "use_npc": True,
    },
    {
        "name":       "No NPC Pass",
        "use_local":  False, "use_rag": True,
        "use_memory": True,  "use_npc": False,
    },
    {
        "name":       "Bare DM Only",
        "use_local":  False, "use_rag": False,
        "use_memory": False, "use_npc": False,
    },
    {
        "name":       "Local Fine-tuned",
        "use_local":  True,  "use_rag": True,
        "use_memory": True,  "use_npc": True,
    },
    {
        "name":       "Local FT + No RAG",
        "use_local":  True,  "use_rag": False,
        "use_memory": True,  "use_npc": True,
    },
]

# ---------------------------------------------------------------------------
# Test Inputs — curated set covering all input types
# ---------------------------------------------------------------------------
TEST_INPUTS = [
    # Combat
    {"input": "I draw my sword and attack the goblin chief.",                        "type": "combat"},
    {"input": "I cast Fireball at the cluster of skeletons near the doorway.",       "type": "combat"},
    {"input": "I try to grapple the orc and pin him against the wall.",              "type": "combat"},
    {"input": "I shoot my crossbow at the bandit on the rooftop.",                   "type": "combat"},
    {"input": "I dodge behind a pillar to avoid the dragon's fire breath.",          "type": "combat"},
    # Roleplay / Social
    {"input": "I tell Durnan I'm looking for information about the Undermountain.",  "type": "roleplay"},
    {"input": "I try to persuade the guard to let us through without paying.",       "type": "roleplay"},
    {"input": "I intimidate the merchant into giving us a discount.",                "type": "roleplay"},
    {"input": "I ask the innkeeper if anyone suspicious has been asking about us.",  "type": "roleplay"},
    # Exploration
    {"input": "I search the dead guard's body for keys or valuables.",               "type": "exploration"},
    {"input": "I carefully examine the runes carved into the dungeon door.",         "type": "exploration"},
    {"input": "I sneak down the corridor to listen at the door.",                    "type": "exploration"},
    {"input": "I climb the rope to reach the ledge above.",                          "type": "exploration"},
    # Rules edge cases
    {"input": "I use my Bonus Action to hide after attacking.",                      "type": "rules"},
    {"input": "Can I cast two spells in the same turn?",                             "type": "rules"},
    {"input": "I'm at 0 HP — what happens next?",                                   "type": "rules"},
    {"input": "I try to disarm the trap using my thieves' tools.",                   "type": "rules"},
    # Dice results
    {"input": "I rolled a 17 on my stealth check.",                                  "type": "dice"},
    {"input": "Natural 20 on my attack roll!",                                       "type": "dice"},
    {"input": "I got a 4 on my persuasion check.",                                   "type": "dice"},
    # Ambiguous / complex
    {"input": "I drink the mysterious glowing potion I found in the chest.",         "type": "complex"},
    {"input": "I try to reason with the dragon instead of fighting it.",             "type": "complex"},
    {"input": "I use my Mage Hand to steal the key from the sleeping guard.",        "type": "complex"},
    {"input": "I attempt to push the statue off the ledge onto the enemies below.",  "type": "complex"},
    {"input": "I tell the party to split up — half go left, half go right.",         "type": "complex"},
]

QUICK_INPUTS = TEST_INPUTS[:8]   # Smoke-test subset

# ---------------------------------------------------------------------------
# LLM-as-Judge Scoring
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You are an expert evaluator of AI Dungeon Master responses for D&D 5e.
Score the DM response on four criteria. Return ONLY a valid JSON object, no explanation."""

JUDGE_PROMPT = """
PLAYER ACTION: "{player_input}"

DM RESPONSE:
{dm_response}

Score each criterion from 1 (poor) to 5 (excellent):

1. narrative_quality   — immersion, creativity, descriptive richness, engagement
2. rules_accuracy      — correct application of D&D 5e mechanics and rulings
3. character_voice     — distinct NPC personality, consistent tone and dialogue
4. relevance           — directly addresses the player's stated action

Return EXACTLY this JSON (integer values only):
{{"narrative_quality": X, "rules_accuracy": X, "character_voice": X, "relevance": X}}
"""

METRICS = ["narrative_quality", "rules_accuracy", "character_voice", "relevance"]


def judge_response(client: Groq, player_input: str, dm_response: str) -> dict:
    """Scores a single DM response using Groq as the judge."""
    prompt = JUDGE_PROMPT.format(
        player_input=player_input,
        dm_response=dm_response[:1500],   # truncate to stay in token budget
    )
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            model=Config.LLM_MODEL['GROQ'],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        scores = json.loads(resp.choices[0].message.content)
        return {m: int(scores.get(m, 3)) for m in METRICS}
    except Exception as e:
        print(f"    [Judge] Scoring failed: {e} — using defaults.")
        return {m: 3 for m in METRICS}


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
def run_evaluation(test_inputs: list[dict]) -> list[dict]:
    """Runs all experiments × all inputs and returns a flat list of result dicts."""
    from orchestrator import RAGnarokOrchestrator

    judge_client = Groq(api_key=Config.GROQ_API_KEY)
    all_results  = []

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"CONDITION: {exp['name']}")
        print(f"{'='*60}")

        # Skip local experiments if Ollama isn't running
        if exp["use_local"]:
            try:
                import requests
                requests.get("http://localhost:11434", timeout=2)
            except Exception:
                print("  [SKIP] Ollama not running. Start with: ollama serve")
                continue

        try:
            system = RAGnarokOrchestrator(
                use_local  = exp["use_local"],
                use_rag    = exp["use_rag"],
                use_memory = exp["use_memory"],
                use_npc    = exp["use_npc"],
                use_voice  = False,
                smart_routing = False,   # disable for fair ablation comparison
            )
        except Exception as e:
            print(f"  [SKIP] Failed to initialize: {e}")
            continue

        exp_scores = []

        for i, test in enumerate(test_inputs):
            player_input = test["input"]
            print(f"  [{i+1}/{len(test_inputs)}] {player_input[:60]}...")

            turn = system.process_turn(player_input)
            dm_response = turn.get("response", "")

            print(f"    Scoring with LLM judge...")
            scores = judge_response(judge_client, player_input, dm_response)
            composite = round(sum(scores.values()) / len(scores), 2)

            result = {
                "experiment":    exp["name"],
                "input":         player_input,
                "input_type":    test["type"],
                "response":      dm_response,
                "latency_ms":    turn.get("latency_ms", 0),
                "skipped_agents": turn.get("skipped", []),
                "scores":        scores,
                "composite":     composite,
            }
            exp_scores.append(result)
            all_results.append(result)

            print(f"    Scores: {scores}  | Composite: {composite} | {turn.get('latency_ms')}ms")
            time.sleep(0.3)   # gentle rate-limiting

        avg = round(sum(r["composite"] for r in exp_scores) / len(exp_scores), 2)
        print(f"\n  → Average composite score for '{exp['name']}': {avg}")

    return all_results


def save_results(results: list[dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {RESULTS_FILE}")


def load_results() -> list[dict]:
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def build_plots(results: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — works without a display
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[Plots] matplotlib not installed: pip install matplotlib")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate by experiment
    from collections import defaultdict
    agg: dict[str, dict] = defaultdict(lambda: {
        "composite": [], "latency": [], **{m: [] for m in METRICS}
    })
    for r in results:
        exp = r["experiment"]
        agg[exp]["composite"].append(r["composite"])
        agg[exp]["latency"].append(r["latency_ms"])
        for m in METRICS:
            agg[exp][m].append(r["scores"][m])

    experiments = list(agg.keys())
    avg_composite = [round(sum(agg[e]["composite"]) / len(agg[e]["composite"]), 2) for e in experiments]
    avg_latency   = [round(sum(agg[e]["latency"])   / len(agg[e]["latency"]),   0) for e in experiments]
    avg_metrics   = {m: [round(sum(agg[e][m]) / len(agg[e][m]), 2) for e in experiments] for m in METRICS}

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]
    short_names = [e.replace("Pipeline ", "").replace("Fine-tuned", "FT") for e in experiments]

    # ------------------------------------------------------------------ #
    # Plot 1 — Average composite score per configuration
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_names, avg_composite, color=COLORS[:len(experiments)], width=0.6, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Composite Score (1–5)", fontsize=11)
    ax.set_title("Average LLM-Judge Score by Configuration", fontsize=13, fontweight="bold")
    ax.axhline(avg_composite[0], color="gray", linestyle="--", linewidth=0.8, label="Baseline")
    ax.legend(fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scores_avg.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR / 'scores_avg.png'}")

    # ------------------------------------------------------------------ #
    # Plot 2 — Grouped bar chart: per-metric breakdown
    # ------------------------------------------------------------------ #
    n_exp     = len(experiments)
    n_metrics = len(METRICS)
    x         = np.arange(n_exp)
    width     = 0.18
    metric_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    metric_labels = ["Narrative Quality", "Rules Accuracy", "Character Voice", "Relevance"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color, label) in enumerate(zip(METRICS, metric_colors, metric_labels)):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, avg_metrics[metric], width, label=label, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (1–5)", fontsize=11)
    ax.set_title("Per-Metric Scores by Configuration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scores_grouped.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR / 'scores_grouped.png'}")

    # ------------------------------------------------------------------ #
    # Plot 3 — Latency comparison
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_names, avg_latency, color=COLORS[:len(experiments)], width=0.6, edgecolor="white")
    ax.bar_label(bars, fmt="%dms", padding=3, fontsize=9)
    ax.set_ylabel("Average Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Latency by Configuration", fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR / 'latency.png'}")

    # ------------------------------------------------------------------ #
    # Plot 4 — Radar chart (top 4 configs by composite score)
    # ------------------------------------------------------------------ #
    top_indices = sorted(range(n_exp), key=lambda i: avg_composite[i], reverse=True)[:4]
    top_exps    = [experiments[i] for i in top_indices]

    categories = metric_labels
    N = len(categories)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    radar_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for idx, (exp, color) in enumerate(zip(top_exps, radar_colors)):
        values = [avg_metrics[m][experiments.index(exp)] for m in METRICS]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=exp)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_title("Radar: Top 4 Configurations", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "radar.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR / 'radar.png'}")

    # ------------------------------------------------------------------ #
    # Plot 5 — Score heatmap (config × metric)
    # ------------------------------------------------------------------ #
    try:
        import numpy as np
        matrix = np.array([[avg_metrics[m][i] for m in METRICS] for i in range(n_exp)])

        fig, ax = plt.subplots(figsize=(8, max(4, n_exp * 0.7)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels(short_names, fontsize=9)
        for i in range(n_exp):
            for j in range(n_metrics):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        fontsize=9, color="black")
        plt.colorbar(im, ax=ax, label="Score (1–5)")
        ax.set_title("Score Heatmap: Configuration × Metric", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved → {PLOTS_DIR / 'heatmap.png'}")
    except Exception as e:
        print(f"  [Heatmap] Skipped: {e}")

    # ------------------------------------------------------------------ #
    # Plot 6 — Score vs Latency scatter (quality-speed trade-off)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (exp, color) in enumerate(zip(experiments, COLORS)):
        ax.scatter(avg_latency[i], avg_composite[i], s=120, color=color,
                   label=short_names[i], zorder=3)
        ax.annotate(short_names[i], (avg_latency[i], avg_composite[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Average Latency (ms)", fontsize=11)
    ax.set_ylabel("Composite Score (1–5)", fontsize=11)
    ax.set_title("Quality vs. Latency Trade-off", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "quality_vs_latency.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR / 'quality_vs_latency.png'}")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Summary table printed to console
# ---------------------------------------------------------------------------
def print_summary(results: list[dict]):
    from collections import defaultdict
    agg = defaultdict(list)
    lat = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r["composite"])
        lat[r["experiment"]].append(r["latency_ms"])

    print("\n" + "=" * 75)
    print(f"{'Configuration':<28} {'Avg Score':>10} {'Avg Latency':>14} {'Samples':>8}")
    print("-" * 75)
    for exp, scores in agg.items():
        avg_s = sum(scores) / len(scores)
        avg_l = sum(lat[exp]) / len(lat[exp])
        print(f"{exp:<28} {avg_s:>10.2f} {avg_l:>13.0f}ms {len(scores):>8}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGnarok Evaluation Harness")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip evaluation, re-generate plots from existing results file")
    parser.add_argument("--quick",     action="store_true",
                        help="Run only 8 test inputs for a quick smoke test")
    args = parser.parse_args()

    if args.plot_only:
        if not RESULTS_FILE.exists():
            print(f"ERROR: {RESULTS_FILE} not found. Run without --plot-only first.")
            sys.exit(1)
        print("Loading existing results...")
        results = load_results()
    else:
        inputs = QUICK_INPUTS if args.quick else TEST_INPUTS
        print(f"Running evaluation: {len(EXPERIMENTS)} configurations × {len(inputs)} inputs")
        print(f"Total LLM calls (approx): {len(EXPERIMENTS) * len(inputs) * 2}\n")
        results = run_evaluation(inputs)
        save_results(results)

    print_summary(results)

    print("\nGenerating plots...")
    build_plots(results)

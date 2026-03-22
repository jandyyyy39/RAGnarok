"""
evaluate.py — RAGnarok Ablation Study Evaluation Harness

This script is designed for experimentation:
- Run ablation configurations one-by-one (to avoid Groq rate limits)
- Save results per configuration to separate JSON files in `data/`
- Resume from partially-completed runs

It computes local metrics (reference-based + reference-free) and optional
LLM-as-judge scoring.

Usage:
  python scripts/evaluate.py --experiment "No Memory"         # run one config, save per-config results
  python scripts/evaluate.py --experiment "No RAG"            # run one config
  python scripts/evaluate.py                                    # full run (all configs) -> data/eval_results.json
  python scripts/evaluate.py --quick                           # 8 inputs (smoke test)
  python scripts/evaluate.py --plot-only                        # plots from data/eval_results.json
  python scripts/evaluate.py --experiment "No Memory" --plot-only
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config

DATA_DIR      = ROOT / "data"
RESULTS_FILE  = DATA_DIR / "eval_results.json"
PLOTS_DIR     = DATA_DIR / "plots"
FIREBALL_EVAL = Config.EVAL_FILE

# Ablation study configurations — each removes or varies one dimension
EXPERIMENTS = [
    {"name": "Full Pipeline (Groq)",  "use_local": False, "use_rag": True,  "use_memory": True,  "use_npc": True},
    {"name": "No RAG",                "use_local": False, "use_rag": False, "use_memory": True,  "use_npc": True},
    {"name": "No Memory",             "use_local": False, "use_rag": True,  "use_memory": False, "use_npc": True},
    {"name": "No NPC Pass",           "use_local": False, "use_rag": True,  "use_memory": True,  "use_npc": False},
    {"name": "Bare DM Only",          "use_local": False, "use_rag": False, "use_memory": False, "use_npc": False},
    {"name": "Local FT (RAG On)",    "use_local": True,  "use_rag": True,  "use_memory": True,  "use_npc": True},
    {"name": "Local FT (No RAG)",    "use_local": True,  "use_rag": False, "use_memory": True,  "use_npc": True},
]

def experiment_slug(name: str) -> str:
    """
    Turn an experiment display name into a filesystem-safe slug.
    Example: "No Memory" -> "no_memory"
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

RULE_KEYWORDS = [
    "roll", "check", "saving throw", "dc", "attack", "damage", "hit points",
    "ability", "modifier", "advantage", "disadvantage", "proficiency",
    "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma",
    "action", "bonus action", "reaction", "spell", "armor class",
]

LLM_JUDGE_METRICS = ["narrative_quality", "rules_accuracy", "character_voice", "relevance"]
LLM_JUDGE_SYSTEM = "You are an expert evaluator of AI Dungeon Master responses for D&D 5e. Score each criterion 1-5. Return ONLY valid JSON."
LLM_JUDGE_PROMPT = '''
PLAYER ACTION: "{player_input}"

DM RESPONSE:
{dm_response}

Score 1-5: narrative_quality, rules_accuracy, character_voice, relevance.
Return JSON: {{"narrative_quality": N, "rules_accuracy": N, "character_voice": N, "relevance": N}}
'''


# ---------------------------------------------------------------------------
# Quantitative metrics (local, no API)
# ---------------------------------------------------------------------------

def compute_rouge(hypothesis: str, reference: str) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {"rouge1": round(scores["rouge1"].fmeasure, 4), "rouge2": round(scores["rouge2"].fmeasure, 4), "rougeL": round(scores["rougeL"].fmeasure, 4)}
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore(hypothesis: str, reference: str) -> float:
    try:
        from bert_score import score as bert_score
        import warnings
        warnings.filterwarnings("ignore")
        _, _, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
        return round(F1[0].item(), 4)
    except ImportError:
        return 0.0


def compute_rule_coverage(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in RULE_KEYWORDS if kw in text_lower)
    return round(hits / len(RULE_KEYWORDS), 4)


def _tokenize_for_diversity(text: str) -> list[str]:
    # Simple tokenizer for distinct-n; avoids punctuation exploding the n-gram space.
    return re.findall(r"\w+", text.lower())


def compute_distinct_n(text: str, n: int) -> float:
    """
    Distinct-n measures output diversity.
    Why it matters: when components are removed, models can become more repetitive.
    """
    toks = _tokenize_for_diversity(text)
    if len(toks) < n or n <= 0:
        return 0.0
    ngrams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return round(len(set(ngrams)) / max(len(ngrams), 1), 4)


def _extract_rule_keyword_hits(text: str) -> set[str]:
    text_lower = text.lower()
    return {kw for kw in RULE_KEYWORDS if kw in text_lower}


def compute_rule_keyword_precision_recall_f1(hypothesis: str, reference: str) -> dict:
    """
    Mechanics-grounding metric vs. reference:
    - Precision: rule terms found in the DM response
    - Recall: how many reference rule terms are covered by the response
    - F1 balances both
    """
    hyp_hits = _extract_rule_keyword_hits(hypothesis)
    ref_hits = _extract_rule_keyword_hits(reference)
    tp = len(hyp_hits.intersection(ref_hits))
    prec = tp / len(hyp_hits) if hyp_hits else 0.0
    rec = tp / len(ref_hits) if ref_hits else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "rule_kw_precision": round(prec, 4),
        "rule_kw_recall": round(rec, 4),
        "rule_kw_f1": round(f1, 4),
    }


def compute_bleu(hypothesis: str, reference: str) -> float:
    """
    BLEU (precision-oriented n-gram overlap) complements ROUGE (recall-oriented).
    Output is normalized to 0–1.
    """
    try:
        from sacrebleu.metrics import BLEU
        metric = BLEU(effective_order=True)
        score = metric.sentence_score(hypothesis, [reference]).score  # 0–100
        return round(score / 100.0, 4)
    except Exception:
        return 0.0


def compute_chrf(hypothesis: str, reference: str) -> float:
    """
    chrF captures character n-gram similarity (often more forgiving than word-level BLEU).
    Output is normalized to 0–1.
    """
    try:
        from sacrebleu.metrics import CHRF
        metric = CHRF(word_order=2)
        score = metric.sentence_score(hypothesis, [reference]).score  # 0–100
        return round(score / 100.0, 4)
    except Exception:
        return 0.0


def compute_meteor(hypothesis: str, reference: str) -> float:
    """
    (deprecated) METEOR is intentionally no longer used in the ablation outputs.
    Kept only to avoid breaking older code paths.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        return round(float(meteor_score([reference], hypothesis)), 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Qualitative metrics (LLM-as-judge, requires Groq API)
# ---------------------------------------------------------------------------

def llm_judge(player_input: str, dm_response: str, use_llm: bool) -> dict:
    if not use_llm:
        return {m: None for m in LLM_JUDGE_METRICS}
    try:
        from groq import Groq
        client = Groq(api_key=Config.GROQ_API_KEY)
        prompt = LLM_JUDGE_PROMPT.format(player_input=player_input, dm_response=dm_response[:1500])
        resp = client.chat.completions.create(
            messages=[{"role": "system", "content": LLM_JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            model=Config.LLM_MODEL['GROQ'],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        scores = json.loads(resp.choices[0].message.content)
        return {m: int(scores.get(m, 3)) for m in LLM_JUDGE_METRICS}
    except Exception as e:
        print(f"    [LLM Judge] Failed: {e}")
        return {m: None for m in LLM_JUDGE_METRICS}


# ---------------------------------------------------------------------------
# Load test set
# ---------------------------------------------------------------------------

def _infer_input_type(text: str) -> str:
    """Heuristic classification for per-type breakdown when FIREBALL has no type field."""
    t = text.lower()
    if any(w in t for w in ["roll", "natural 20", "natural 1", "d20", "dice", "check", "saving throw"]):
        return "dice"
    if any(w in t for w in ["attack", "cast", "fireball", "spell", "grapple", "hit", "damage", "sword"]):
        return "combat"
    if any(w in t for w in ["persuade", "tell", "ask", "say", "convince", "intimidate", "deceive"]):
        return "roleplay"
    if any(w in t for w in ["can i", "how does", "rule", "bonus action", "reaction", "two spells"]):
        return "rules"
    if any(w in t for w in ["search", "examine", "look", "inspect", "open", "door", "room"]):
        return "exploration"
    return "general"


def load_test_inputs(limit: int = 25) -> list[dict]:
    if FIREBALL_EVAL.exists():
        samples = []
        with open(FIREBALL_EVAL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    itype = row.get("type") or _infer_input_type(row.get("input", ""))
                    samples.append({"input": row["input"], "reference": row["output"], "type": itype})
                if len(samples) >= limit:
                    break
        print(f"Loaded {len(samples)} test inputs from FIREBALL eval set.")
        return samples
    print(f"WARNING: {FIREBALL_EVAL} not found. Using fallback inputs (no reference for ROUGE/BERTScore).")
    return FALLBACK_INPUTS[:limit]


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
    {"input": "I try to reason with the dragon instead of fighting it.",         "reference": "", "type": "complex"},
]


# ---------------------------------------------------------------------------
# Evaluation runner (resume + per-config outputs)
# ---------------------------------------------------------------------------

def _result_key_for_skip(exp_name: str, test: dict) -> str:
    # Inputs are stable across runs; reference length included to reduce collisions.
    ref = test.get("reference", "") or ""
    return f"{exp_name}||{test.get('type','unknown')}||{len(ref)}||{test.get('input','')}"


def _load_results_if_exists(out_file: Path) -> list[dict]:
    if not out_file.exists():
        return []
    try:
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_results(out_file: Path, results: list[dict]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_experiment_to_file(
    test_inputs: list[dict],
    exp: dict,
    use_llm_judge: bool,
    out_file: Path,
    resume: bool = True,
    extra_metrics_mode: str = "all",
) -> list[dict]:
    """
    Run one ablation config and write partial results after each input.
    This prevents losing progress when Groq hits 429.
    """
    from orchestrator import RAGnarokOrchestrator

    existing = []
    if resume:
        # Prefer per-config file (from previous resumed runs).
        existing = _load_results_if_exists(out_file)
        # Backward-compat: reuse legacy combined quick results (`data/eval_results.json`)
        # when the per-config file doesn't exist yet.
        if not existing and RESULTS_FILE.exists():
            try:
                with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                    legacy = json.load(f)
                existing = [r for r in legacy if r.get("experiment") == exp["name"]]
            except Exception:
                existing = []
    existing_keys = set()
    for r in existing:
        existing_keys.add(
            f"{r.get('experiment')}||{r.get('input_type','unknown')}||{len(r.get('reference','') or '')}||{r.get('input','')}"
        )

    all_results = list(existing)

    print(f"\n{'='*60}\nCONDITION: {exp['name']}\n{'='*60}")

    if exp["use_local"]:
        try:
            import requests
            requests.get("http://localhost:11434", timeout=2)
        except Exception:
            print("  [SKIP] Ollama not running. Start with: ollama serve")
            return all_results

    system = RAGnarokOrchestrator(
        use_local=exp["use_local"],
        use_rag=exp["use_rag"],
        use_memory=exp["use_memory"],
        use_npc=exp["use_npc"],
        smart_routing=False,
    )

    # Deterministic cost proxy: expected number of Groq LLM calls per turn.
    # - Rules Arbiter only runs when use_rag=True
    # - DM Agent only calls Groq when use_local=False
    # - NPC Consistency runs when use_npc=True
    # - LLM-as-judge is an extra Groq call when enabled
    expected_groq_calls = (
        (1 if exp.get("use_rag") else 0)
        + (1 if not exp.get("use_local") else 0)
        + (1 if exp.get("use_npc") else 0)
        + (1 if use_llm_judge else 0)
    )

    for i, test in enumerate(test_inputs):
        player_input = test["input"]
        reference = test.get("reference", "")
        input_type = test.get("type", "unknown")

        key = _result_key_for_skip(exp["name"], test)
        if resume and key in existing_keys:
            continue

        print(f"  [{i+1}/{len(test_inputs)}] {player_input[:55]}...")

        try:
            turn = system.process_turn(player_input)
            hypothesis = turn.get("response", "")
        except Exception as e:
            msg = str(e)
            print(f"    [ERROR] {type(e).__name__}: {msg}")
            _save_results(out_file, all_results)
            if "rate_limit_exceeded" in msg or "429" in msg.lower():
                print("    [STOP] Groq rate limit reached. Partial results saved; rerun later with resume.")
            else:
                print("    [STOP] Generation error. Partial results saved.")
            return all_results

        scores = {}

        compute_bleu_chrf = extra_metrics_mode in ("all", "bleu_chrf_only")
        # METEOR and rule keyword P/R/F1 are intentionally excluded from the ablation outputs.
        compute_meteor_and_rule_kw = False
        compute_distinct = extra_metrics_mode == "all"
        if reference:
            rouge = compute_rouge(hypothesis, reference)
            scores.update(rouge)
            scores["bertscore"] = compute_bertscore(hypothesis, reference)

            # BLEU + chrF are classic overlap metrics that complement ROUGE/BERTScore.
            # In the final ablation we keep them, because they are cheap and informative.
            if compute_bleu_chrf:
                scores["bleu"] = compute_bleu(hypothesis, reference)
                scores["chrf"] = compute_chrf(hypothesis, reference)

            # METEOR and rule keyword P/R/F1 are excluded.
        else:
            scores["rouge1"] = None
            scores["rouge2"] = None
            scores["rougeL"] = None
            scores["bertscore"] = None
            if compute_bleu_chrf:
                scores["bleu"] = None
                scores["chrf"] = None

        # Response-based metrics (always computable)
        scores["rule_coverage"] = compute_rule_coverage(hypothesis)
        scores["response_length"] = len(hypothesis.split())
        if compute_distinct:
            scores["distinct1"] = compute_distinct_n(hypothesis, 1)
            scores["distinct2"] = compute_distinct_n(hypothesis, 2)

        # Optional LLM judge (qualitative, expensive; use only when you can)
        llm_scores = llm_judge(player_input, hypothesis, use_llm_judge)
        if use_llm_judge and any(v is not None for v in llm_scores.values()):
            scores["llm_composite"] = round(sum(v for v in llm_scores.values() if v is not None) / 4, 2)
            scores.update(llm_scores)
            time.sleep(0.2)
        else:
            scores["llm_composite"] = None
            scores.update({m: None for m in LLM_JUDGE_METRICS})

        result = {
            "experiment": exp["name"],
            "input": player_input,
            "input_type": input_type,
            "response": hypothesis,
            "reference": reference,
            "latency_ms": turn.get("latency_ms", 0),
            "safety_pass": turn.get("safety_pass", None),
            "skipped": turn.get("skipped", []),
            "expected_groq_calls": expected_groq_calls,
            "scores": scores,
        }

        all_results.append(result)
        existing_keys.add(key)

        # Save after each input: makes the process restartable.
        _save_results(out_file, all_results)

        q = f"ROUGE-L={scores.get('rougeL')} " if scores.get("rougeL") is not None else ""
        j = f"LLM={scores.get('llm_composite')} " if scores.get("llm_composite") else ""
        print(f"    {q}{j}RuleCov={scores['rule_coverage']} {turn.get('latency_ms')}ms")

    return all_results

def run_evaluation(test_inputs: list[dict], use_llm_judge: bool) -> list[dict]:
    from orchestrator import RAGnarokOrchestrator

    all_results = []
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}\nCONDITION: {exp['name']}\n{'='*60}")

        if exp["use_local"]:
            try:
                import requests
                requests.get("http://localhost:11434", timeout=2)
            except Exception:
                print("  [SKIP] Ollama not running. Start with: ollama serve")
                continue

        try:
            system = RAGnarokOrchestrator(use_local=exp["use_local"], use_rag=exp["use_rag"],
                use_memory=exp["use_memory"], use_npc=exp["use_npc"], smart_routing=False)
        except Exception as e:
            print(f"  [SKIP] Init failed: {e}")
            continue

        for i, test in enumerate(test_inputs):
            player_input = test["input"]
            reference   = test.get("reference", "")
            print(f"  [{i+1}/{len(test_inputs)}] {player_input[:55]}...")

            turn       = system.process_turn(player_input)
            hypothesis = turn.get("response", "")

            scores = {}
            if reference:
                rouge = compute_rouge(hypothesis, reference)
                scores.update(rouge)
                scores["bertscore"] = compute_bertscore(hypothesis, reference)
            else:
                scores["rouge1"] = scores["rouge2"] = scores["rougeL"] = scores["bertscore"] = None
            scores["rule_coverage"] = compute_rule_coverage(hypothesis)
            scores["response_length"] = len(hypothesis.split())

            llm_scores = llm_judge(player_input, hypothesis, use_llm_judge)
            if use_llm_judge and any(v is not None for v in llm_scores.values()):
                scores["llm_composite"] = round(sum(v for v in llm_scores.values() if v is not None) / 4, 2)
                scores.update(llm_scores)
                time.sleep(0.2)
            else:
                scores["llm_composite"] = None
                scores.update({m: None for m in LLM_JUDGE_METRICS})

            result = {
                "experiment": exp["name"], "input": player_input, "input_type": test.get("type", "unknown"),
                "response": hypothesis, "reference": reference, "latency_ms": turn.get("latency_ms", 0),
                "skipped": turn.get("skipped", []), "scores": scores,
            }
            all_results.append(result)
            q = f"ROUGE-L={scores.get('rougeL')} " if scores.get('rougeL') is not None else ""
            j = f"LLM={scores.get('llm_composite')} " if scores.get('llm_composite') else ""
            print(f"    {q}{j}RuleCov={scores['rule_coverage']} {turn.get('latency_ms')}ms")

    return all_results


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        exp = r["experiment"]
        agg[exp]["latency"].append(r["latency_ms"])
        for k, v in r["scores"].items():
            if v is not None:
                agg[exp][k].append(v)

    def avg(exp, k): return round(sum(agg[exp][k]) / len(agg[exp][k]), 3) if agg[exp].get(k) else "-"
    def lat(exp): return round(sum(agg[exp]["latency"]) / len(agg[exp]["latency"])) if agg[exp]["latency"] else 0

    cols = ["rougeL", "bertscore", "rule_coverage", "llm_composite", "response_length", "latency_ms"]
    has_llm = any(agg[e].get("llm_composite") for e in agg)
    if not has_llm:
        cols = [c for c in cols if c != "llm_composite"]

    print("\n" + "=" * 100)
    header = f"{'Configuration':<28}"
    for c in cols:
        if c == "latency_ms":
            header += f" {'Latency':>10}"
        else:
            header += f" {c[:10]:>10}"
    print(header)
    print("-" * 100)
    for exp in agg:
        row = f"{exp:<28}"
        for c in cols:
            if c == "latency_ms":
                row += f" {lat(exp):>9}ms"
            else:
                row += f" {str(avg(exp, c)):>10}"
        print(row)
    print("=" * 100)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def build_plots(results: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Plots] pip install matplotlib numpy")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = defaultdict(lambda: defaultdict(list))
    by_type = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in results:
        exp, itype = r["experiment"], r["input_type"]
        agg[exp]["latency"].append(r["latency_ms"])
        for k, v in r["scores"].items():
            if v is not None:
                agg[exp][k].append(v)
                by_type[exp][itype][k].append(v)

    experiments = list(agg.keys())
    short_names = [e.replace("Full Pipeline ", "Full\n").replace("Local FT", "FT").replace(" (Groq)", "") for e in experiments]
    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]

    def avg(exp, k): return round(sum(agg[exp][k]) / len(agg[exp][k]), 3) if agg[exp].get(k) else 0
    avg_lat = [round(sum(agg[e]["latency"]) / max(len(agg[e]["latency"]), 1)) for e in experiments]
    has_llm = any(agg[e].get("llm_composite") for e in experiments)
    plot_metrics = ["rouge1", "rouge2", "rougeL", "bertscore", "rule_coverage"]
    if has_llm:
        plot_metrics = plot_metrics + ["llm_composite"] + LLM_JUDGE_METRICS
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "Rule Cov"] + (["LLM Composite"] + ["Narr Q", "Rules Acc", "Char Voice", "Relevance"] if has_llm else [])
    n_exp, n_met = len(experiments), len(plot_metrics)

    # 1. Grouped quantitative metrics (normalize LLM 1–5 to 0–1 for consistency)
    def norm_val(v, m):
        if v is None or v == 0: return 0
        if any(x in m for x in ["llm", "narrative", "rules", "character", "relevance"]):
            return v / 5.0
        return v

    x = np.arange(n_exp)
    width = 0.12
    met_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#DA8BC3", "#937860", "#2CA02C", "#FF7F0E", "#1F77B4"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (m, color) in enumerate(zip(plot_metrics, met_colors)):
        vals = [norm_val(avg(e, m), m) for e in experiments]
        offset = (i - n_met / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=metric_labels[i] if i < len(metric_labels) else m, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_title("Quantitative & Qualitative Metrics by Configuration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scores_grouped.png", dpi=150)
    plt.close()
    print("  Saved -> scores_grouped.png")

    # 2. Ablation impact — delta from baseline (Full Pipeline)
    baseline = experiments[0] if experiments else None
    if baseline:
        baseline_rouge = avg(baseline, "rougeL")
        deltas = [avg(e, "rougeL") - baseline_rouge for e in experiments]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]
        bars = ax.bar(short_names, deltas, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
        ax.set_ylabel("Δ ROUGE-L vs. Baseline", fontsize=11)
        ax.set_title("Ablation Impact: Change in Quality When Removing Components", fontsize=13, fontweight="bold")
        plt.xticks(rotation=15, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ablation_impact.png", dpi=150)
        plt.close()
        print("  Saved -> ablation_impact.png")

    # 3. Per input-type breakdown (if we have multiple types)
    input_types = set()
    for exp in by_type:
        input_types.update(by_type[exp].keys())
    input_types = [t for t in sorted(input_types) if t != "unknown"]
    if len(input_types) >= 2:
        fig, axes = plt.subplots(1, min(4, len(input_types)), figsize=(4 * min(4, len(input_types)), 5), squeeze=False)
        for idx, itype in enumerate(input_types[:4]):
            ax = axes[0, idx]
            vals = []
            for exp in experiments:
                v = sum(by_type[exp][itype]["rougeL"]) / len(by_type[exp][itype]["rougeL"]) if by_type[exp][itype]["rougeL"] else 0
                vals.append(v)
            ax.bar(range(len(experiments)), vals, color=COLORS[:n_exp], edgecolor="white")
            ax.set_xticks(range(len(experiments)))
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("ROUGE-L")
            ax.set_title(f"By input type: {itype}")
            ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "scores_by_input_type.png", dpi=150)
        plt.close()
        print("  Saved -> scores_by_input_type.png")

    # 4. Latency comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_names, avg_lat, color=COLORS[:n_exp], width=0.6, edgecolor="white")
    ax.bar_label(bars, fmt="%dms", padding=3, fontsize=9)
    ax.set_ylabel("Average Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Latency by Configuration", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency.png", dpi=150)
    plt.close()
    print("  Saved -> latency.png")

    # 5. Quality vs. Latency scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (exp, color) in enumerate(zip(experiments, COLORS)):
        ax.scatter(avg_lat[i], avg(exp, "rougeL"), s=140, color=color, label=short_names[i], zorder=3)
        ax.annotate(short_names[i], (avg_lat[i], avg(exp, "rougeL")), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Average Latency (ms)", fontsize=11)
    ax.set_ylabel("ROUGE-L F1", fontsize=11)
    ax.set_title("Quality vs. Latency Trade-off (Pareto Frontier)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "quality_vs_latency.png", dpi=150)
    plt.close()
    print("  Saved -> quality_vs_latency.png")

    # 6. Heatmap
    plot_m = [m for m in plot_metrics if m in ["rouge1", "rouge2", "rougeL", "bertscore", "rule_coverage", "llm_composite"] and any(agg[e].get(m) for e in experiments)]
    if plot_m:
        matrix = np.array([[avg(e, m) for m in plot_m] for e in experiments])
        fig, ax = plt.subplots(figsize=(10, max(4, n_exp * 0.7)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(plot_m)))
        ax.set_xticklabels([m.replace("_", " ").title() for m in plot_m], fontsize=10)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels(short_names, fontsize=9)
        for i in range(n_exp):
            for j in range(len(plot_m)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label="Score (0–1)")
        ax.set_title("Score Heatmap: Configuration × Metric", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "heatmap.png", dpi=150)
        plt.close()
        print("  Saved -> heatmap.png")

    # 7. Radar (top 4)
    top4 = sorted(experiments, key=lambda e: avg(e, "rougeL"), reverse=True)[:4]
    rad_metrics = ["rougeL", "bertscore", "rule_coverage"]
    if has_llm:
        rad_metrics.append("llm_composite")
    N = len(rad_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for exp, color in zip(top4, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
        vals = [avg(exp, m) / (5.0 if "llm" in m else 1.0) for m in rad_metrics] + [avg(exp, rad_metrics[0]) / (5.0 if "llm" in rad_metrics[0] else 1.0)]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=exp)
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in rad_metrics], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Radar: Top 4 Configurations by ROUGE-L", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "radar.png", dpi=150)
    plt.close()
    print("  Saved -> radar.png")

    # 8. Agent contribution stacked (ablation from full)
    if baseline and n_exp >= 5:
        contrib = []
        labels = []
        full_score = avg(baseline, "rougeL")
        for exp in ["No RAG", "No Memory", "No NPC Pass", "Bare DM Only"]:
            if exp in agg:
                s = avg(exp, "rougeL")
                contrib.append(full_score - s)
                labels.append(exp.replace("No ", "-").replace(" ", "\n"))
        if contrib:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(labels, contrib, color=["#C44E52", "#DD8452", "#937860", "#8172B2"], edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Quality Drop (ROUGE-L) vs. Full Pipeline", fontsize=11)
            ax.set_title("Contribution of Each Agent: How Much Quality Is Lost When Removed", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "agent_contribution.png", dpi=150)
            plt.close()
            print("  Saved -> agent_contribution.png")

    # 9. LLM-as-judge breakdown (if available)
    if has_llm and all(agg[e].get("llm_composite") for e in experiments):
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(n_exp)
        w = 0.15
        for i, m in enumerate(LLM_JUDGE_METRICS):
            vals = [avg(e, m) for e in experiments]
            ax.bar(x + (i - 2) * w, vals, w, label=m.replace("_", " ").title(), edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Score (1–5)", fontsize=11)
        ax.set_title("LLM-as-Judge: Qualitative Metrics by Configuration", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "llm_judge_breakdown.png", dpi=150)
        plt.close()
        print("  Saved -> llm_judge_breakdown.png")

    # 10. Configuration ranking (sorted by ROUGE-L)
    rank_metric = "llm_composite" if has_llm else "rougeL"
    ranked = sorted(experiments, key=lambda e: avg(e, rank_metric) or 0, reverse=True)
    rank_vals = [avg(e, rank_metric) or 0 for e in ranked]
    rank_names = [e.replace("Full Pipeline (Groq)", "Full").replace("Local FT + No RAG", "FT+NoRAG") for e in ranked]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(ranked)), rank_vals, color=[COLORS[ranked.index(e) % len(COLORS)] for e in ranked], edgecolor="white")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(rank_names, fontsize=9)
    ax.set_xlabel(rank_metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Configuration Ranking by {rank_metric.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "config_ranking.png", dpi=150)
    plt.close()
    print("  Saved -> config_ranking.png")

    # 11. Response length distribution by config
    len_by_exp = {e: agg[e]["response_length"] for e in experiments if agg[e].get("response_length")}
    if len_by_exp:
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [len_by_exp[e] for e in experiments]
        bp = ax.boxplot(data, labels=short_names, patch_artist=True, vert=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
        ax.set_ylabel("Response Length (words)", fontsize=11)
        ax.set_title("Response Length Distribution by Configuration", fontsize=13, fontweight="bold")
        plt.xticks(rotation=15, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "response_length_dist.png", dpi=150)
        plt.close()
        print("  Saved -> response_length_dist.png")

    # 12. Metric correlation matrix (across all samples)
    num_metrics = ["rougeL", "bertscore", "rule_coverage", "response_length", "latency_ms"]
    if has_llm:
        num_metrics = num_metrics + ["llm_composite"]
    rows = []
    for r in results:
        row = []
        valid = True
        for m in num_metrics:
            if m == "latency_ms":
                v = r.get("latency_ms", 0)
            else:
                v = r["scores"].get(m)
            if v is None:
                valid = False
                break
            row.append(v)
        if valid and len(row) == len(num_metrics):
            rows.append(row)
    if len(rows) >= 10:
        mat = np.array(rows)
        corr = np.corrcoef(mat.T)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(num_metrics)))
        ax.set_yticks(range(len(num_metrics)))
        ax.set_xticklabels([m.replace("_", "\n") for m in num_metrics], fontsize=9)
        ax.set_yticklabels([m.replace("_", "\n") for m in num_metrics], fontsize=9)
        for i in range(len(num_metrics)):
            for j in range(len(num_metrics)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Metric Correlation Matrix (across samples)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "metric_correlation.png", dpi=150)
        plt.close()
        print("  Saved -> metric_correlation.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGnarok Ablation Study Evaluation")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate plots from an existing results JSON")
    parser.add_argument("--quick", action="store_true", help="Use only 8 test inputs")
    parser.add_argument("--llm-judge", action="store_true", help="Add LLM-as-judge scoring (requires Groq API)")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM-as-judge (default)")
    parser.add_argument("--n", type=int, default=25, help="Number of test inputs")

    # Experiment-level execution (recommended for Groq rate limit workflows)
    parser.add_argument("--experiment", type=str, default=None, help="Run only one ablation config (name or slug).")
    parser.add_argument("--out-prefix", type=str, default="eval_results", help="Output prefix in data/: eval_results_<slug>.json")
    parser.add_argument("--fresh", action="store_true", help="Overwrite the selected experiment output file (no resume).")
    parser.add_argument(
        "--extra-metrics-mode",
        type=str,
        default="all",
        choices=["all", "bleu_chrf_only", "none"],
        help="Controls which reference-based overlap metrics + generation diversity metrics are written to outputs.",
    )
    parser.add_argument("--no-extra-metrics", action="store_true", help="Alias for --extra-metrics-mode none.")

    args = parser.parse_args()
    use_llm = args.llm_judge and not args.no_llm_judge

    limit = 8 if args.quick else args.n
    inputs = load_test_inputs(limit=limit)

    # ---- Plot-only mode ----
    if args.plot_only:
        if args.experiment:
            exp_query = args.experiment.strip().lower()
            matches = []
            for e in EXPERIMENTS:
                if e["name"].lower() == exp_query or experiment_slug(e["name"]).startswith(exp_query) or exp_query in e["name"].lower():
                    matches.append(e)
            if not matches:
                print(f"ERROR: No experiment matches --experiment={args.experiment}")
                sys.exit(1)
            if len(matches) > 1:
                print("ERROR: --plot-only with a broad --experiment query is ambiguous. Use exact name or slug.")
                sys.exit(1)
            per_out = DATA_DIR / f"{args.out_prefix}_{experiment_slug(matches[0]['name'])}.json"
            if not per_out.exists():
                print(f"ERROR: {per_out} not found. Run the experiment first.")
                sys.exit(1)
            with open(per_out, "r", encoding="utf-8") as f:
                results = json.load(f)
            print_summary(results)
            print("\nGenerating plots...")
            build_plots(results)
            sys.exit(0)

        # No experiment specified -> plot the legacy combined file
        if not RESULTS_FILE.exists():
            print(f"ERROR: {RESULTS_FILE} not found. Run without --plot-only first.")
            sys.exit(1)
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print_summary(results)
        print("\nGenerating plots...")
        build_plots(results)
        sys.exit(0)

    # ---- Full run (legacy behavior) ----
    if not args.experiment:
        print(f"\nAblation Study: {len(EXPERIMENTS)} configs × {len(inputs)} inputs")
        print("Quantitative: ROUGE/BERTScore + rule coverage + latency (local)")
        if use_llm:
            print("Qualitative: LLM-as-judge enabled\n")
        else:
            print("Qualitative: disabled (use --llm-judge to enable)\n")

        results = run_evaluation(inputs, use_llm_judge=use_llm)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {RESULTS_FILE}")
        print_summary(results)
        print("\nGenerating plots...")
        build_plots(results)
        sys.exit(0)

    # ---- Experiment-only run (recommended) ----
    exp_query = args.experiment.strip().lower()
    matches = []
    for e in EXPERIMENTS:
        if e["name"].lower() == exp_query or experiment_slug(e["name"]).startswith(exp_query) or exp_query in e["name"].lower():
            matches.append(e)
    if not matches:
        print(f"ERROR: No experiment matches --experiment={args.experiment}. Try one of:")
        for e in EXPERIMENTS:
            print(f"  - {e['name']} (slug: {experiment_slug(e['name'])})")
        sys.exit(1)
    if len(matches) > 1:
        print(f"WARNING: multiple experiments match '{args.experiment}'. Running {len(matches)} configs sequentially.")

    all_loaded = []
    for e in matches:
        per_out = DATA_DIR / f"{args.out_prefix}_{experiment_slug(e['name'])}.json"
        print(f"\n[Output] {per_out}")
        if args.fresh and per_out.exists():
            per_out.unlink()

        extra_mode = "none" if args.no_extra_metrics else args.extra_metrics_mode
        loaded = run_experiment_to_file(
            test_inputs=inputs,
            exp=e,
            use_llm_judge=use_llm,
            out_file=per_out,
            resume=not args.fresh,
            extra_metrics_mode=extra_mode,
        )
        all_loaded.extend(loaded)

    if all_loaded:
        print_summary(all_loaded)

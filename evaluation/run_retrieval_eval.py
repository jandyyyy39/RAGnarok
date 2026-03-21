"""
Retrieval-Only Evaluation
=========================
Tests ONLY the retrieval step of each RAG method. No LLM ruling calls.
Runs in minutes (except HyDE/CRAG which still call LLM for their internal
generate/grade steps — use -p local to run those via Ollama instead of Groq).

Relevance definition:
  A chunk is relevant if it contains >= 1 primary keyword AND >= 1 supporting keyword.
  - primary_keywords: the specific SRD mechanic/skill that MUST appear (e.g. "grapple")
  - supporting_keywords: ability name or SRD context (e.g. "athletics", "contested")
  This prevents generic ability-score words ("charisma", "wisdom") from inflating
  precision by matching unrelated skill descriptions.

Metrics:
  - Precision@k   : fraction of k retrieved chunks that are relevant
  - Recall@k      : fraction of primary keywords found across all retrieved chunks
  - MRR           : 1 / rank of first relevant chunk
  - NDCG@k        : discounted cumulative gain (rewards relevant docs ranked higher)
  - F1@k          : harmonic mean of Precision@k and Recall@k
  - Hit@k         : 1 if any relevant chunk was retrieved, else 0
  - MAP           : mean average precision (rewards both relevance and rank order)

Usage:
    python evaluation/run_retrieval_eval.py              # uses Groq API
    python evaluation/run_retrieval_eval.py -p local     # uses local Ollama model
    python evaluation/run_retrieval_eval.py -p fast      # uses Groq fast model

Output:
    evaluation/results/retrieval_summary.json
    evaluation/results/retrieval_summary.csv
    evaluation/results/retrieval_naive.json
    evaluation/results/retrieval_hyde.json
    evaluation/results/retrieval_hybrid.json
    evaluation/results/retrieval_crag.json
"""

import argparse
import json
import math
import time
import sys
import os
import csv
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from openai import OpenAI
from groq import Groq
from agents.rules_arbiter import RulesArbiter
from agents.rules_arbiter_hyde import RulesArbiterHyDE
from agents.rules_arbiter_hybrid import RulesArbiterHybrid
from agents.rules_arbiter_crag import RulesArbiterCRAG

DB_PATH = "data/chroma_db"
WORLD_CTX = "Location: The Black Boar Tavern. Active NPCs: Thrain Blackbeard."


# ═══════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════

def chunk_is_relevant(chunk_text, test_case):
    """
    A chunk is relevant if it contains:
      - at least 1 primary keyword (the specific SRD mechanic — MUST appear)
      - at least 1 supporting keyword (if any are defined)

    This prevents generic terms like "charisma" or "wisdom" from inflating
    precision by matching unrelated skill descriptions.
    """
    chunk_lower = chunk_text.lower()
    primary = test_case.get("primary_keywords", [])
    supporting = test_case.get("supporting_keywords", [])

    primary_match = any(kw.lower() in chunk_lower for kw in primary)

    if supporting:
        supporting_match = any(kw.lower() in chunk_lower for kw in supporting)
    else:
        supporting_match = True

    return primary_match and supporting_match


def precision_at_k(chunks, test_case):
    """Of k retrieved chunks, what fraction are relevant?"""
    if not chunks:
        return 0.0
    hits = sum(1 for c in chunks if chunk_is_relevant(c, test_case))
    return hits / len(chunks)


def recall_at_k(chunks, test_case):
    """
    Fraction of primary keywords found across all retrieved chunks.
    We use primary_keywords only since supporting keywords are SRD-guaranteed;
    primary keywords are the core concepts we must retrieve.
    """
    primary = test_case.get("primary_keywords", [])
    if not primary:
        return 0.0
    all_text = ' '.join(chunks).lower()
    found = sum(1 for kw in primary if kw.lower() in all_text)
    return found / len(primary)


def mrr_score(chunks, test_case):
    """1 / rank of the first relevant chunk. 0 if none are relevant."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_is_relevant(chunk, test_case):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunks, test_case):
    """
    Normalized Discounted Cumulative Gain at k.
    Rewards methods that place relevant chunks at higher ranks.
    Better than MRR because it rewards multiple relevant chunks, not just rank-1.
    """
    relevance = [1 if chunk_is_relevant(c, test_case) else 0 for c in chunks]

    # DCG: sum of rel_i / log2(rank + 1)
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))

    # Ideal DCG: all relevant docs ranked first
    n_relevant = sum(relevance)
    ideal = [1] * n_relevant + [0] * (len(chunks) - n_relevant)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def f1_at_k(precision, recall):
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_at_k(chunks, test_case):
    """Binary: 1 if at least one relevant chunk was retrieved, else 0."""
    return 1.0 if any(chunk_is_relevant(c, test_case) for c in chunks) else 0.0


def average_precision(chunks, test_case):
    """
    Average Precision for a single query.
    Sums precision at each rank where a relevant doc is found,
    then normalises by the total number of relevant docs retrieved.
    Used to compute MAP across all queries.
    """
    hits = 0
    precision_sum = 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_is_relevant(chunk, test_case):
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / hits


# ═══════════════════════════════════════
# RETRIEVAL FUNCTIONS (per method)
# ═══════════════════════════════════════

def retrieve_naive(arbiter, query):
    """Naive: direct similarity search, k=5 (same as all other methods)."""
    docs = arbiter.vectorstore.similarity_search(query, k=5)
    return [d.page_content for d in docs]


def retrieve_with_method(arbiter, query):
    """HyDE / Hybrid / CRAG: use the .retrieve() method."""
    docs = arbiter.retrieve(query, WORLD_CTX)
    return [d.page_content for d in docs]


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="RAGnarok Retrieval Evaluation")
    parser.add_argument(
        "-p", "--profile",
        choices=["local", "fast", "groq"],
        default="groq",
        help="local = Ollama (no API cost), groq = Groq API, fast = Groq fast model"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["naive", "hyde", "hybrid", "crag", "all"],
        default="all",
        help="Which retrieval method to evaluate (default: all)"
    )
    args = parser.parse_args()

    # --- Load test cases ---
    with open('evaluation/test_cases.json') as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} retrieval test cases\n")

    os.makedirs('evaluation/results', exist_ok=True)

    # --- Initialize client based on profile ---
    if args.profile == "local":
        print("Using local Ollama model:", Config.LLM_MODEL['LOCAL'])
        print("Make sure Ollama is running: ollama serve")
        print("Make sure model is pulled:   ollama pull", Config.LLM_MODEL['LOCAL'], "\n")
        client = OpenAI(base_url=Config.OLLAMA_BASE_URL, api_key="ollama")
        model_profile = "LOCAL"
    elif args.profile == "fast":
        print("Using Groq fast model:", Config.LLM_MODEL['GROQ_FAST'])
        client = Groq(api_key=Config.GROQ_API_KEY)
        model_profile = "GROQ_FAST"
    else:
        print("Using Groq API model:", Config.LLM_MODEL['GROQ'])
        client = Groq(api_key=Config.GROQ_API_KEY)
        model_profile = "GROQ"

    # Note: Naive and Hybrid do NOT call the LLM during retrieval.
    # The client is only used by HyDE (1 call/query) and CRAG (up to 12 calls/query).

    # --- Determine which methods to run ---
    selected = ['naive', 'hyde', 'hybrid', 'crag'] if args.method == 'all' else [args.method]

    all_method_classes = {
        'naive':  ('naive',  lambda: RulesArbiter(client, model_profile, db_path=DB_PATH)),
        'hyde':   ('hyde',   lambda: RulesArbiterHyDE(client, model_profile, db_path=DB_PATH)),
        'hybrid': ('hybrid', lambda: RulesArbiterHybrid(client, model_profile, db_path=DB_PATH)),
        'crag':   ('crag',   lambda: RulesArbiterCRAG(client, model_profile, db_path=DB_PATH)),
    }

    print(f"Initializing method(s): {', '.join(selected)}")
    methods = {k: (name, factory()) for k, (name, factory) in all_method_classes.items() if k in selected}
    print("Ready.\n")

    # --- Run evaluation ---
    all_summaries = []
    metric_keys = ['precision', 'recall', 'mrr', 'ndcg', 'f1', 'hit', 'ap']

    for method_key, (method_name, arbiter) in methods.items():
        print(f"{'='*60}")
        print(f"  Evaluating {method_name.upper()}")
        print(f"{'='*60}")

        results = []
        scores = defaultdict(list)
        by_category = defaultdict(lambda: defaultdict(list))
        by_query_type = defaultdict(lambda: defaultdict(list))

        for tc in test_cases:
            print(f"  [{tc['id']:3d}] {tc['query'][:45]:<45}", end=" ", flush=True)

            start = time.time()

            try:
                if method_name == 'naive':
                    chunks = retrieve_naive(arbiter, tc['query'])
                else:
                    chunks = retrieve_with_method(arbiter, tc['query'])
            except Exception as e:
                print(f"ERROR: {e}")
                chunks = []

            latency = time.time() - start

            # Compute all metrics
            p = precision_at_k(chunks, tc)
            r = recall_at_k(chunks, tc)
            m = mrr_score(chunks, tc)
            n = ndcg_at_k(chunks, tc)
            f = f1_at_k(p, r)
            h = hit_at_k(chunks, tc)
            a = average_precision(chunks, tc)

            for key, val in zip(metric_keys, [p, r, m, n, f, h, a]):
                scores[key].append(val)
                by_category[tc['category']][key].append(val)
                by_query_type[tc['query_type']][key].append(val)

            results.append({
                'id': tc['id'],
                'query': tc['query'],
                'category': tc['category'],
                'query_type': tc['query_type'],
                'primary_keywords': tc.get('primary_keywords', []),
                'supporting_keywords': tc.get('supporting_keywords', []),
                'num_chunks_retrieved': len(chunks),
                'chunks_preview': [c[:150] for c in chunks],
                'precision': round(p, 4),
                'recall': round(r, 4),
                'mrr': round(m, 4),
                'ndcg': round(n, 4),
                'f1': round(f, 4),
                'hit': int(h),
                'ap': round(a, 4),
                'latency': round(latency, 3)
            })

            status = "HIT" if h else "MISS"
            print(f"P={p:.2f} R={r:.2f} MRR={m:.2f} NDCG={n:.2f} F1={f:.2f} AP={a:.2f} [{status}] ({latency:.1f}s)")

            # Throttle only when hitting external APIs (not needed for local Ollama)
            if method_name in ('hyde', 'crag') and model_profile != 'LOCAL':
                time.sleep(1)

        # Save per-method raw results
        with open(f'evaluation/results/retrieval_{method_name}.json', 'w') as f_out:
            json.dump(results, f_out, indent=2)

        # Compute averages
        avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0

        cat_summary = {}
        for cat, s in by_category.items():
            cat_summary[cat] = {k: avg(v) for k, v in s.items()}
            cat_summary[cat]['n'] = len(s['precision'])

        qt_summary = {}
        for qt, s in by_query_type.items():
            qt_summary[qt] = {k: avg(v) for k, v in s.items()}
            qt_summary[qt]['n'] = len(s['precision'])

        summary = {
            'method': method_name,
            'model_profile': model_profile,
            'n_queries': len(test_cases),
            'avg_precision_at_k': avg(scores['precision']),
            'avg_recall_at_k': avg(scores['recall']),
            'avg_mrr': avg(scores['mrr']),
            'avg_ndcg': avg(scores['ndcg']),
            'avg_f1': avg(scores['f1']),
            'avg_hit_at_k': avg(scores['hit']),
            'map': avg(scores['ap']),
            'by_category': cat_summary,
            'by_query_type': qt_summary,
        }
        all_summaries.append(summary)

        print(f"\n  Precision@k : {summary['avg_precision_at_k']}")
        print(f"  Recall@k    : {summary['avg_recall_at_k']}")
        print(f"  MRR         : {summary['avg_mrr']}")
        print(f"  NDCG@k      : {summary['avg_ndcg']}")
        print(f"  F1@k        : {summary['avg_f1']}")
        print(f"  Hit@k       : {summary['avg_hit_at_k']}")
        print(f"  MAP         : {summary['map']}")
        print()

    # --- Save combined summary ---
    with open('evaluation/results/retrieval_summary.json', 'w') as f_out:
        json.dump(all_summaries, f_out, indent=2)

    fields = ['method', 'model_profile', 'avg_precision_at_k', 'avg_recall_at_k',
              'avg_mrr', 'avg_ndcg', 'avg_f1', 'avg_hit_at_k', 'map', 'n_queries']
    with open('evaluation/results/retrieval_summary.csv', 'w', newline='') as f_out:
        w = csv.DictWriter(f_out, fieldnames=fields)
        w.writeheader()
        for s in all_summaries:
            w.writerow({k: s[k] for k in fields})

    print(f"{'='*60}")
    print(f"Saved: evaluation/results/retrieval_summary.json")
    print(f"Saved: evaluation/results/retrieval_summary.csv")
    print(f"\nNext: python evaluation/generate_retrieval_charts.py")


if __name__ == '__main__':
    main()

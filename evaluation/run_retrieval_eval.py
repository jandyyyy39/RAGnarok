import argparse
import json
import math
import time
import os, sys
import csv
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config
from openai import OpenAI
from groq import Groq
from agents.rules_arbiter import RulesArbiter
from agents.rules_arbiter_hyde import RulesArbiterHyDE
from agents.rules_arbiter_hybrid import RulesArbiterHybrid
from agents.rules_arbiter_crag import RulesArbiterCRAG

from scripts.test_rag import (
    test_retrieval_cosine_sim,
    test_bm25,
    retrieve_with_rerank,
    retrieve_bm25_with_rerank,
    build_bm25_index,
)

DB_PATH = "data/chroma_db"
WORLD_CTX = "Location: The Black Boar Tavern. Active NPCs: Thrain Blackbeard."

# -------------- Helper functions (General) --------------
def normalize_chunks(results):
    if not results:
        return []

    normalized = []
    for r in results:
        if isinstance(r, str):
            normalized.append(r)
        elif isinstance(r, dict):
            if "text" in r:
                normalized.append(r["text"])
            elif "doc" in r and hasattr(r["doc"], "page_content"):
                normalized.append(r["doc"].page_content)
            else:
                normalized.append(str(r))
        elif hasattr(r, "page_content"):
            normalized.append(r.page_content)
        else:
            normalized.append(str(r))
    return normalized

def retrieve_naive_arbiter_eval(arbiter, query):
    docs = arbiter.vectorstore.similarity_search(query, k=5)
    return normalize_chunks(docs)

def retrieve_arbiter_method_eval(arbiter, query, world_ctx):
    docs = arbiter.retrieve(query, world_ctx)
    return normalize_chunks(docs)

def chunk_is_relevant(chunk_text, test_case):
    chunk_lower = chunk_text.lower()
    primary = test_case.get("primary_keywords", [])
    supporting = test_case.get("supporting_keywords", [])

    primary_match = any(kw.lower() in chunk_lower for kw in primary)

    if supporting:
        supporting_match = any(kw.lower() in chunk_lower for kw in supporting)
    else:
        supporting_match = True

    return primary_match and supporting_match

# -------------- Metrics Functions --------------
def precision_at_k(chunks, test_case):
    if not chunks:
        return 0.0
    hits = sum(1 for c in chunks if chunk_is_relevant(c, test_case))
    return hits / len(chunks)

def recall_at_k(chunks, test_case):
    primary = test_case.get("primary_keywords", [])
    if not primary:
        return 0.0
    all_text = " ".join(chunks).lower()
    found = sum(1 for kw in primary if kw.lower() in all_text)
    return found / len(primary)

def mrr_score(chunks, test_case):
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_is_relevant(chunk, test_case):
            return 1.0 / rank
    return 0.0

def ndcg_at_k(chunks, test_case):
    relevance = [1 if chunk_is_relevant(c, test_case) else 0 for c in chunks]
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))

    n_relevant = sum(relevance)
    ideal = [1] * n_relevant + [0] * (len(chunks) - n_relevant)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0

def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def hit_at_k(chunks, test_case):
    return 1.0 if any(chunk_is_relevant(c, test_case) for c in chunks) else 0.0

def average_precision(chunks, test_case):
    hits = 0
    precision_sum = 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_is_relevant(chunk, test_case):
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / hits

# -------------- Helper Functions (RAG Systems) --------------
def retrieve_cosine_eval(query):
    return normalize_chunks(test_retrieval_cosine_sim(query))

def retrieve_bm25_eval(bm25, final_splits, query):
    return normalize_chunks(test_bm25(bm25, final_splits, query))

def retrieve_cosine_rerank_eval(query):
    return normalize_chunks(retrieve_with_rerank(query))

def retrieve_bm25_rerank_eval(bm25, final_splits, query):
    return normalize_chunks(retrieve_bm25_with_rerank(bm25, final_splits, query))

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
    metric_keys = ["precision", "recall", "mrr", "ndcg", "f1", "hit", "ap"]

    for method_key, (method_name, arbiter) in methods.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating {method_name.upper()}")
        print(f"{'='*60}")

        results = []
        scores = defaultdict(list)

        for tc in test_cases:
            start = time.time()
            try:
                chunks = normalize_chunks(arbiter.retrieve(tc["query"], WORLD_CTX))
            except Exception as e:
                print(f"ERROR on {method_name} / query {tc['id']}: {e}")
                chunks = []

            latency = time.time() - start

            p = precision_at_k(chunks, tc)
            r = recall_at_k(chunks, tc)
            m = mrr_score(chunks, tc)
            n = ndcg_at_k(chunks, tc)
            f1 = f1_at_k(p, r)
            h = hit_at_k(chunks, tc)
            ap = average_precision(chunks, tc)

            for key, val in zip(metric_keys, [p, r, m, n, f1, h, ap]):
                scores[key].append(val)

            results.append({
                "id": tc["id"],
                "query": tc["query"],
                "category": tc.get("category"),
                "query_type": tc.get("query_type"),
                "num_chunks_retrieved": len(chunks),
                "chunks_preview": [c[:150] for c in chunks],
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mrr": round(m, 4),
                "ndcg": round(n, 4),
                "f1": round(f1, 4),
                "hit": int(h),
                "ap": round(ap, 4),
                "latency": round(latency, 3),
            })

            print(
                f"[{tc['id']:3d}] "
                f"P={p:.2f} R={r:.2f} MRR={m:.2f} "
                f"NDCG={n:.2f} F1={f1:.2f} AP={ap:.2f}"
            )

        with open(f"evaluation/results_v2/retrieval_{method_name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0

        summary = {
            "method": method_name,
            "n_queries": len(test_cases),
            "avg_precision_at_k": avg(scores["precision"]),
            "avg_recall_at_k": avg(scores["recall"]),
            "avg_mrr": avg(scores["mrr"]),
            "avg_ndcg": avg(scores["ndcg"]),
            "avg_f1": avg(scores["f1"]),
            "avg_hit_at_k": avg(scores["hit"]),
            "map": avg(scores["ap"]),
        }

        all_summaries.append(summary)

        print(f"\n--- {method_name.upper()} SUMMARY ({len(test_cases)} queries) ---")
        print(f"  Precision@5 : {summary['avg_precision_at_k']:.4f}")
        print(f"  Recall@5    : {summary['avg_recall_at_k']:.4f}")
        print(f"  MRR         : {summary['avg_mrr']:.4f}")
        print(f"  NDCG@5      : {summary['avg_ndcg']:.4f}")
        print(f"  F1@5        : {summary['avg_f1']:.4f}")
        print(f"  Hit@5       : {summary['avg_hit_at_k']:.4f}")
        print(f"  MAP         : {summary['map']:.4f}")

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    header = f"{'Method':<10} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6} {'F1':>6} {'Hit@5':>6} {'MAP':>6}"
    print(header)
    print("-" * 70)
    for s in all_summaries:
        print(
            f"{s['method']:<10} "
            f"{s['avg_precision_at_k']:>6.4f} "
            f"{s['avg_recall_at_k']:>6.4f} "
            f"{s['avg_mrr']:>6.4f} "
            f"{s['avg_ndcg']:>6.4f} "
            f"{s['avg_f1']:>6.4f} "
            f"{s['avg_hit_at_k']:>6.4f} "
            f"{s['map']:>6.4f}"
        )

    with open("evaluation/results_v2/retrieval_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    fields = [
        "method",
        "avg_precision_at_k",
        "avg_recall_at_k",
        "avg_mrr",
        "avg_ndcg",
        "avg_f1",
        "avg_hit_at_k",
        "map",
        "n_queries",
    ]

    with open("evaluation/results_v2/retrieval_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_summaries:
            writer.writerow({k: row[k] for k in fields})

if __name__ == "__main__":
    main()
import json
import math
import time
import os
import sys
import csv
from collections import defaultdict

# Make imports work regardless of execution directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from config import Config
from groq import Groq
from openai import OpenAI
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

DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
TEST_CASES_PATH = os.path.join(BASE_DIR, "evaluation", "rag", "test_cases.json")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "rag", "results_v2")
WORLD_CTX = "Location: The Black Boar Tavern. Active NPCs: Thrain Blackbeard."
RESULTS_DIR = "evaluation/results_v2"

MODEL_PROFILE_FOR_SUMMARY = "LOCAL"

# -------------- Helper functions (General) --------------
def normalize_chunks(results):
    """
    Convert retrieval outputs into plain chunk text for content-based evaluation.

    Supported input shapes:
      - str
      - LangChain Document-like objects with .page_content
      - dicts containing:
          * "text": str
          * "doc": Document-like object with .page_content

    Unknown shapes are skipped instead of being stringified.
    """
    if not results:
        return []

    normalized = []

    for r in results:
        if isinstance(r, str):
            text = r

        elif hasattr(r, "page_content"):
            text = r.page_content

        elif isinstance(r, dict):
            if isinstance(r.get("text"), str):
                text = r["text"]
            elif "doc" in r and hasattr(r["doc"], "page_content"):
                text = r["doc"].page_content
            else:
                continue

        else:
            continue

        if text is None:
            continue

        text = str(text).strip()
        if text:
            normalized.append(text)

    return normalized

def retrieve_arbiter_method_eval(arbiter, query, world_ctx):
    docs = arbiter.retrieve(query, world_ctx)
    return normalize_chunks(docs)

# -------------- Relevance / Metric functions --------------
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

# -------------- Retrieval wrappers --------------
def retrieve_cosine_eval(query):
    return normalize_chunks(test_retrieval_cosine_sim(query))

def retrieve_bm25_eval(bm25, final_splits, query):
    return normalize_chunks(test_bm25(bm25, final_splits, query))

def retrieve_cosine_rerank_eval(query):
    return normalize_chunks(retrieve_with_rerank(query))

def retrieve_bm25_rerank_eval(bm25, final_splits, query):
    return normalize_chunks(retrieve_bm25_with_rerank(bm25, final_splits, query))

# -------------- Summary rebuild helpers --------------
def safe_avg(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0

def summarise_group(rows):
    return {
        "precision": safe_avg([r.get("precision", 0.0) for r in rows]),
        "recall": safe_avg([r.get("recall", 0.0) for r in rows]),
        "mrr": safe_avg([r.get("mrr", 0.0) for r in rows]),
        "ndcg": safe_avg([r.get("ndcg", 0.0) for r in rows]),
        "f1": safe_avg([r.get("f1", 0.0) for r in rows]),
        "hit": safe_avg([r.get("hit", 0.0) for r in rows]),
        "ap": safe_avg([r.get("ap", 0.0) for r in rows]),
        "n": len(rows),
    }

def group_rows(rows, key_name):
    grouped = defaultdict(list)

    for row in rows:
        key = row.get(key_name)
        if key is not None and str(key).strip():
            grouped[str(key)].append(row)

    return {
        group_name: summarise_group(group)
        for group_name, group in grouped.items()
    }

def rebuild_summary_from_individual_results(results_dir, model_profile):
    summaries = []

    for filename in sorted(os.listdir(results_dir)):
        if not filename.startswith("retrieval_") or not filename.endswith(".json"):
            continue
        if filename == "retrieval_summary.json":
            continue

        path = os.path.join(results_dir, filename)

        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        if not rows:
            continue

        raw_method_name = filename[len("retrieval_"):-len(".json")]

        summary = {
            "method": raw_method_name,
            "model_profile": model_profile,
            "n_queries": len(rows),
            "avg_precision_at_k": safe_avg([r.get("precision", 0.0) for r in rows]),
            "avg_recall_at_k": safe_avg([r.get("recall", 0.0) for r in rows]),
            "avg_mrr": safe_avg([r.get("mrr", 0.0) for r in rows]),
            "avg_ndcg": safe_avg([r.get("ndcg", 0.0) for r in rows]),
            "avg_f1": safe_avg([r.get("f1", 0.0) for r in rows]),
            "avg_hit_at_k": safe_avg([r.get("hit", 0.0) for r in rows]),
            "map": safe_avg([r.get("ap", 0.0) for r in rows]),
            "by_category": group_rows(rows, "category"),
            "by_query_type": group_rows(rows, "query_type"),
        }

        summaries.append(summary)

    return summaries

def write_summary_csv(summaries, output_csv_path):
    fields = [
        "method",
        "model_profile",
        "n_queries",
        "avg_precision_at_k",
        "avg_recall_at_k",
        "avg_mrr",
        "avg_ndcg",
        "avg_f1",
        "avg_hit_at_k",
        "map",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in summaries:
            writer.writerow({k: row.get(k) for k in fields})

def main():
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Building BM25 index...")
    bm25, final_splits = build_bm25_index()

    # client = Groq(api_key=Config.GROQ_API_KEY)
    # runtime_model_profile = "GROQ"

    client = OpenAI(base_url=Config.OLLAMA_BASE_URL, api_key="ollama")
    runtime_model_profile = "LOCAL"

    hyde_arbiter = RulesArbiterHyDE(client, runtime_model_profile, db_path=DB_PATH)
    hybrid_arbiter = RulesArbiterHybrid(client, runtime_model_profile, db_path=DB_PATH)
    crag_arbiter = RulesArbiterCRAG(client, runtime_model_profile, db_path=DB_PATH)

    methods = {
        "cosine": lambda q: retrieve_cosine_eval(q),
        "bm25": lambda q: retrieve_bm25_eval(bm25, final_splits, q),
        "cosine_rerank": lambda q: retrieve_cosine_rerank_eval(q),
        "bm25_rerank": lambda q: retrieve_bm25_rerank_eval(bm25, final_splits, q),
        "hyde": lambda q: retrieve_arbiter_method_eval(hyde_arbiter, q, WORLD_CTX),
        "hybrid": lambda q: retrieve_arbiter_method_eval(hybrid_arbiter, q, WORLD_CTX),
        "crag": lambda q: retrieve_arbiter_method_eval(crag_arbiter, q, WORLD_CTX),
    }

    for method_name, retrieve_fn in methods.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {method_name.upper()}")
        print(f"{'=' * 60}")

        results = []

        for tc in test_cases:
            start = time.time()
            try:
                chunks = retrieve_fn(tc["query"])
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

        output_path = os.path.join(RESULTS_DIR, f"retrieval_{method_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Rebuild summary from saved per-query files so schema stays consistent
    summaries = rebuild_summary_from_individual_results(
        results_dir=RESULTS_DIR,
        model_profile=MODEL_PROFILE_FOR_SUMMARY,
    )

    with open("evaluation/results_v2/retrieval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

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
        for row in summaries:
            writer.writerow({k: row[k] for k in fields})


if __name__ == "__main__":
    main()
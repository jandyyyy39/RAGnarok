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
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    bm25, final_splits = build_bm25_index()

    client = Groq(api_key=Config.GROQ_API_KEY)
    model_profile = "GROQ"

    naive_arbiter = RulesArbiter(client, model_profile, db_path=DB_PATH)
    hyde_arbiter = RulesArbiterHyDE(client, model_profile, db_path=DB_PATH)
    hybrid_arbiter = RulesArbiterHybrid(client, model_profile, db_path=DB_PATH)
    crag_arbiter = RulesArbiterCRAG(client, model_profile, db_path=DB_PATH)

    methods = {
        "cosine": lambda q: retrieve_cosine_eval(q),
        "bm25": lambda q: retrieve_bm25_eval(bm25, final_splits, q),
        "cosine_rerank": lambda q: retrieve_cosine_rerank_eval(q),
        "bm25_rerank": lambda q: retrieve_bm25_rerank_eval(bm25, final_splits, q),
        "naive_old": lambda q: retrieve_naive_arbiter_eval(naive_arbiter, q),
        "hyde": lambda q: retrieve_arbiter_method_eval(hyde_arbiter, q, WORLD_CTX),
        "hybrid": lambda q: retrieve_arbiter_method_eval(hybrid_arbiter, q, WORLD_CTX),
        "crag": lambda q: retrieve_arbiter_method_eval(crag_arbiter, q, WORLD_CTX),
    }

    all_summaries = []
    metric_keys = ["precision", "recall", "mrr", "ndcg", "f1", "hit", "ap"]

    for method_name, retrieve_fn in methods.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {method_name.upper()}")
        print(f"{'=' * 60}")

        results = []
        scores = defaultdict(list)

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

        with open(os.path.join(RESULTS_DIR, f"retrieval_{method_name}.json"), "w", encoding="utf-8") as f:
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

    with open(os.path.join(RESULTS_DIR, "retrieval_summary.json"), "w", encoding="utf-8") as f:
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

    with open(os.path.join(RESULTS_DIR, "retrieval_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv..DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_summaries:
            writer.writerow({k: row[k] for k in fields})


if __name__ == "__main__":
    main()
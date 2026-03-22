import json
import re
import csv
import time
import os
import sys
from collections import defaultdict
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import Config
from groq import Groq

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

EVAL_SET_PATH = "./data/eval_set.json"
DB_PATH = "data/chroma_db"
WORLD_CTX = "Location: The Black Boar Tavern. Active NPCs: Thrain Blackbeard."
K = 5

# ── Normalisation helpers ─────────────────────────────────────────────────────
def strip_anchor(text: str | None) -> str:
    """Remove {#section-anchor} style fragments from header strings."""
    if text is None:
        return ""
    return re.sub(r"\{#[^}]+\}", "", str(text)).strip()

def normalise(text: str | None) -> str:
    """Lowercase, strip anchors and surrounding whitespace."""
    return strip_anchor(text).lower().strip()

# ── Metadata extraction ────────────────────────────────────────────────────────
def extract_metadata(result: Any) -> dict:
    """
    Handle multiple return shapes:
      - LangChain Document
      - dict with 'metadata'
      - dict with 'doc' (where doc may be a Document)
      - anything else -> {}
    """
    if result is None:
        return {}

    if isinstance(result, dict):
        if isinstance(result.get("metadata"), dict):
            return result["metadata"]

        doc = result.get("doc")
        if doc is not None and isinstance(getattr(doc, "metadata", None), dict):
            return doc.metadata or {}

        return {}

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return metadata or {}

    return {}

def get_deepest_header(metadata: dict) -> str:
    """Return the most specific non-empty header from a chunk metadata dict."""
    for level in ("Header 4", "Header 3", "Header 2", "Header 1"):
        value = metadata.get(level)
        if value is not None and str(value).strip():
            return str(value)
    return ""

def get_all_headers(metadata: dict) -> list[str]:
    """
    Return all header values in the chain normalised.
    A retrieved chunk is relevant if ANY level matches a gold section.
    """
    headers = []
    for level in ("Header 1", "Header 2", "Header 3", "Header 4"):
        value = metadata.get(level)
        if value is not None and str(value).strip():
            headers.append(normalise(value))
    return headers

def get_retrieved_headers(results: list) -> list[str]:
    """
    Extract deepest header from each result for logging/output.
    """
    headers = []
    for r in results:
        metadata = extract_metadata(r)
        raw_header = get_deepest_header(metadata)
        headers.append(normalise(raw_header))
    return headers


def get_header_chain_for_logging(metadata: dict) -> str:
    parts = []
    for level in ("Header 1", "Header 2", "Header 3", "Header 4"):
        value = metadata.get(level)
        if value is not None and str(value).strip():
            parts.append(normalise(value))
    return " > ".join(parts)

# ── Relevance helpers ─────────────────────────────────────────────────────────
def build_relevance_list(results: list, gold_sections: list[str]) -> list[int]:
    """
    Binary relevance list in rank order.
    A result is relevant if ANY header in its full chain matches a gold section.
    Each gold section can only contribute one hit.
    """
    normalised_gold = {normalise(s) for s in gold_sections}
    seen_gold_hits = set()
    relevance = []

    for r in results:
        metadata = extract_metadata(r)
        chain_headers = get_all_headers(metadata)

        matched_gold = next(
            (h for h in chain_headers if h in normalised_gold and h not in seen_gold_hits),
            None
        )

        if matched_gold:
            relevance.append(1)
            seen_gold_hits.add(matched_gold)
        else:
            relevance.append(0)

    return relevance

# ── Metric calculations ───────────────────────────────────────────────────────
def precision_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    """
    Precision@min(k, r), where r is total relevant.
    """
    cutoff = min(k, total_relevant)
    if cutoff == 0:
        return 0.0
    return sum(relevance[:cutoff]) / cutoff

def recall_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant

def reciprocal_rank(relevance: list[int]) -> float:
    for i, rel in enumerate(relevance):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    import math

    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

    actual_dcg = dcg(relevance[:k])
    ideal_rels = [1] * min(total_relevant, k)
    ideal_dcg = dcg(ideal_rels)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def average_precision(relevance: list[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, rel in enumerate(relevance[:k]):
        if rel == 1:
            hits += 1
            precision_sum += hits / (i + 1)

    return precision_sum / total_relevant

def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def hit_at_k(relevance: list[int]) -> float:
    return 1.0 if any(relevance) else 0.0

def compute_metrics(relevance: list[int], total_relevant: int, k: int) -> dict:
    precision = precision_at_k(relevance, k, total_relevant)
    recall = recall_at_k(relevance, k, total_relevant)

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank(relevance),
        "ndcg": ndcg_at_k(relevance, k, total_relevant),
        "f1": f1_at_k(precision, recall),
        "hit": hit_at_k(relevance),
        "ap": average_precision(relevance, total_relevant, k),
    }

# ── Retrieval wrappers ────────────────────────────────────────────────────────
def retrieve_cosine_eval(query):
    return test_retrieval_cosine_sim(query)

def retrieve_bm25_eval(bm25, final_splits, query):
    return test_bm25(bm25, final_splits, query)

def retrieve_cosine_rerank_eval(query):
    return retrieve_with_rerank(query=query)

def retrieve_bm25_rerank_eval(bm25, final_splits, query):
    return retrieve_bm25_with_rerank(bm25, final_splits, query=query)

def retrieve_arbiter_method_eval(arbiter, query, world_ctx):
    """
    Return raw docs/results so header metadata remains available.
    """
    return arbiter.retrieve(query, world_ctx)

# ── Debug helper ──────────────────────────────────────────────────────────────
def print_canonical_headers(final_splits, output_path="canonical_headers.txt"):
    from collections import Counter

    headers = Counter()
    for chunk in final_splits:
        for level in ("Header 1", "Header 2", "Header 3", "Header 4"):
            val = chunk.metadata.get(level)
            if val:
                headers[val.strip()] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for header, count in headers.most_common():
            f.write(f"{count:>4}x  {header}\n")

    print(f"Canonical headers written to {output_path} ({len(headers)} unique headers)")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    os.makedirs("evaluation/results_v2/header_eval", exist_ok=True)

    print("Building BM25 index...")
    bm25, final_splits = build_bm25_index()

    client = Groq(api_key=Config.GROQ_API_KEY)
    model_profile = "GROQ"

    hyde_arbiter = RulesArbiterHyDE(client, model_profile, db_path=DB_PATH)
    hybrid_arbiter = RulesArbiterHybrid(client, model_profile, db_path=DB_PATH)
    crag_arbiter = RulesArbiterCRAG(client, model_profile, db_path=DB_PATH)

    methods = {
        "cosine": lambda q: retrieve_cosine_eval(q),
        "bm25": lambda q: retrieve_bm25_eval(bm25, final_splits, q),
        "cosine_rerank": lambda q: retrieve_cosine_rerank_eval(q),
        "bm25_rerank": lambda q: retrieve_bm25_rerank_eval(bm25, final_splits, q),
        "hyde": lambda q: retrieve_arbiter_method_eval(hyde_arbiter, q, WORLD_CTX),
        "hybrid": lambda q: retrieve_arbiter_method_eval(hybrid_arbiter, q, WORLD_CTX),
        "crag": lambda q: retrieve_arbiter_method_eval(crag_arbiter, q, WORLD_CTX),
    }

    all_summaries = []
    metric_keys = ["precision", "recall", "mrr", "ndcg", "f1", "hit", "ap"]

    crag_debug_printed = False

    for method_name, retrieve_fn in methods.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {method_name.upper()}")
        print(f"{'=' * 60}")

        results = []
        scores = defaultdict(list)

        for entry in eval_set:
            query_id = entry["id"]
            query = entry["player_action"]
            gold_sections = entry["relevant_srd_sections"]
            total_relevant = len(gold_sections)

            start = time.time()
            try:
                retrieved = retrieve_fn(query)
                retrieved = retrieved[:K]
            except Exception as e:
                print(f"ERROR on {method_name} / query {query_id}: {e}")
                retrieved = []

            latency = time.time() - start

            # if method_name == "crag" and retrieved and not crag_debug_printed:
            #     print("\n--- CRAG DEBUG ---")
            #     print("Query ID:", query_id)
            #     print("Query:", query)
            #     print("Gold sections:", gold_sections)
            #     print("Normalised gold:", [normalise(s) for s in gold_sections])
            #     print()

            #     for i, r in enumerate(retrieved[:3], start=1):
            #         md = extract_metadata(r)
            #         print(f"Result {i}")
            #         print("Type:", type(r))
            #         print("Metadata keys:", list(md.keys()))
            #         print("Metadata:", md)
            #         print("Deepest header:", get_deepest_header(md))
            #         print("Header chain:", get_header_chain_for_logging(md))
            #         print("Preview:", getattr(r, "page_content", "")[:200])
            #         print()

            #     print("Computed relevance:", build_relevance_list(retrieved, gold_sections))
            #     print("------------------\n")
            #     crag_debug_printed = True

            retrieved_headers = get_retrieved_headers(retrieved)
            relevance = build_relevance_list(retrieved, gold_sections)
            metrics = compute_metrics(relevance, total_relevant, K)

            for key in metric_keys:
                scores[key].append(metrics[key])

            results.append({
                "id": query_id,
                "query": query,
                "gold_sections": gold_sections,
                "normalised_gold_sections": [normalise(s) for s in gold_sections],
                "num_chunks_retrieved": len(retrieved),
                "retrieved_headers": retrieved_headers,
                "retrieved_header_chains": [
                    get_header_chain_for_logging(extract_metadata(r)) for r in retrieved
                ],
                "relevance_list": relevance,
                "total_relevant": total_relevant,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "mrr": round(metrics["mrr"], 4),
                "ndcg": round(metrics["ndcg"], 4),
                "f1": round(metrics["f1"], 4),
                "hit": int(metrics["hit"]),
                "ap": round(metrics["ap"], 4),
                "latency": round(latency, 3),
            })

            print(
                f"[{query_id:3d}] "
                f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                f"MRR={metrics['mrr']:.2f} NDCG={metrics['ndcg']:.2f} "
                f"F1={metrics['f1']:.2f} AP={metrics['ap']:.2f}"
            )

        with open(f"evaluation/results_v2/header_eval/retrieval_{method_name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0

        summary = {
            "method": method_name,
            "n_queries": len(eval_set),
            "avg_precision_at_k": avg(scores["precision"]),
            "avg_recall_at_k": avg(scores["recall"]),
            "avg_mrr": avg(scores["mrr"]),
            "avg_ndcg": avg(scores["ndcg"]),
            "avg_f1": avg(scores["f1"]),
            "avg_hit_at_k": avg(scores["hit"]),
            "map": avg(scores["ap"]),
        }

        all_summaries.append(summary)

    with open("evaluation/results_v2/header_eval/retrieval_summary.json", "w", encoding="utf-8") as f:
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

    with open("evaluation/results_v2/header_eval/retrieval_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_summaries:
            writer.writerow({k: row[k] for k in fields})

    print_canonical_headers(
        final_splits,
        output_path="evaluation/results_v2/header_eval/canonical_headers.txt",
    )


if __name__ == "__main__":
    main()
import json
import re
import csv
from datetime import datetime
from typing import Any

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from test_rag import (
    test_retrieval_cosine_sim,
    test_bm25,
    retrieve_with_rerank,
    retrieve_bm25_with_rerank,
    build_bm25_index,
)

EVAL_SET_PATH = "./data/eval_set.json"
K = 5

RETRIEVAL_SYSTEMS = [
    "cosine_sim",
    "bm25",
    "cosine_rerank",
    "bm25_rerank",
]

def strip_anchor(text: str | None) -> str:
    """Remove {#section-anchor} style fragments from header strings."""
    if text is None:
        return ""
    return re.sub(r"\{#[^}]+\}", "", str(text)).strip()


def normalise(text: str | None) -> str:
    """Lowercase, strip anchors and surrounding whitespace."""
    return strip_anchor(text).lower().strip()


def get_deepest_header(metadata: dict) -> str:
    """Return the most specific non-empty header from a chunk's metadata dict."""
    for level in ("Header 3", "Header 2", "Header 1"):
        value = metadata.get(level)
        if value is not None and str(value).strip():
            return str(value)
    return ""

def extract_metadata(result: Any) -> dict:
    """
    Normalise the two different return shapes:
      - Document objects  (cosine_sim, bm25)
      - Dicts with 'metadata' key  (reranker variants)
    Always returns the raw metadata dict.
    """
    if isinstance(result, dict):
        return result.get("metadata", {})
    return result.metadata  # LangChain Document

def get_all_headers(metadata: dict) -> list[str]:
    """
    Return all header values in the chain (H1, H2, H3) normalised.
    A retrieved chunk is relevant if ANY level of its header chain
    matches a gold section — not just the deepest level.
    """
    headers = []
    for level in ("Header 1", "Header 2", "Header 3"):
        value = metadata.get(level)
        if value is not None and str(value).strip():
            headers.append(normalise(value))
    return headers

def get_retrieved_headers(results: list) -> list[str]:
    """
    Extract the deepest header from each result for logging/CSV output.
    Returns a list of length K preserving retrieval rank order.
    """
    headers = []
    for r in results:
        metadata = extract_metadata(r)
        raw_header = get_deepest_header(metadata)
        headers.append(normalise(raw_header))
    return headers

def get_header_chain_for_logging(metadata: dict) -> str:
    parts = []
    for level in ("Header 1", "Header 2", "Header 3"):
        value = metadata.get(level)
        if value is not None and str(value).strip():
            parts.append(normalise(value))
    return " > ".join(parts)

# ── Relevance helpers ─────────────────────────────────────────────────────────

def build_relevance_list(results: list, gold_sections: list[str]) -> list[int]:
    """
    Binary relevance list in rank order.
    A result is relevant if ANY header in its full chain matches a gold section.
    Each gold section can only contribute one hit (deduplication).
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
    Precision@min(k, r) where r is the number of relevant items.
    This prevents precision from being artificially capped when r < k.
    """
    cutoff = min(k, total_relevant)

    if cutoff == 0:
        return 0.0

    return sum(relevance[:cutoff]) / cutoff

def recall_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    """Fraction of all relevant sections retrieved in top-K."""
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant

def ndcg_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Ideal DCG is built from total_relevant (all gold sections that exist),
    not just the subset the retriever happened to find.
    """
    import math

    def dcg(rels):
        return sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(rels[:k])
        )

    actual_dcg = dcg(relevance[:k])

    # Ideal: as many 1s as there are gold sections, capped at K
    ideal_rels = [1] * min(total_relevant, k)
    ideal_dcg = dcg(ideal_rels)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def reciprocal_rank(relevance: list[int]) -> float:
    """1 / rank of first relevant result. 0 if none found."""
    for i, rel in enumerate(relevance):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(relevance: list[int], total_relevant: int) -> float:
    """Average precision for a single query."""
    if total_relevant == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, rel in enumerate(relevance):
        if rel == 1:
            hits += 1
            precision_sum += hits / (i + 1)

    return precision_sum / total_relevant

def compute_metrics(relevance: list[int], total_relevant: int, k: int) -> dict:
    # No rounding here — keep full precision for accurate aggregation.
    # Rounding happens only at save/print time.
    return {
        f"precision@{k}": precision_at_k(relevance, k, total_relevant),
        f"recall@{k}":    recall_at_k(relevance, k, total_relevant),
        f"ndcg@{k}":      ndcg_at_k(relevance, k, total_relevant),
        "mrr":            reciprocal_rank(relevance),
        "ap":             average_precision(relevance, total_relevant),
    }

# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_retrieval(system: str, query: str, bm25=None, final_splits=None) -> list:
    """Dispatch query to the correct retrieval function."""
    if system == "cosine_sim":
        return test_retrieval_cosine_sim(query)
    elif system == "bm25":
        return test_bm25(bm25, final_splits, query)
    elif system == "cosine_rerank":
        return retrieve_with_rerank(query=query)
    elif system == "bm25_rerank":
        return retrieve_bm25_with_rerank(bm25, final_splits, query=query)
    else:
        raise ValueError(f"Unknown retrieval system: {system}")

def evaluate(eval_path: str, k: int = K) -> tuple[list[dict], dict]:
    """
    Run all retrieval systems over the full eval set.

    Returns:
        per_query_rows  — one row per (query × system) for the detailed CSV
        aggregate       — mean metrics per system for the summary CSV
    """
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # Build BM25 index once — reused across all BM25 queries
    print("Building BM25 index...")
    bm25, final_splits = build_bm25_index()

    per_query_rows = []

    # Track running totals for aggregate metrics
    agg = {
        system: {
            f"precision@{k}": 0.0,
            f"recall@{k}":    0.0,
            f"ndcg@{k}":      0.0,
            "mrr":            0.0,
            "ap":             0.0,
            "count":          0,
        }
        for system in RETRIEVAL_SYSTEMS
    }

    for entry in eval_set:
        query_id      = entry["id"]
        query         = entry["player_action"]
        gold_sections = entry["relevant_srd_sections"]
        total_relevant = len(gold_sections)

        print(f"\nQuery {query_id}: {query}")

        for system in RETRIEVAL_SYSTEMS:
            print(f"  [{system}]", end=" ")

            try:
                results = run_retrieval(system, query, bm25, final_splits)
            except Exception as e:
                print(f"ERROR: {e}")
                results = []

            retrieved_headers = get_retrieved_headers(results)
            relevance         = build_relevance_list(results, gold_sections)
            metrics           = compute_metrics(relevance, total_relevant, k)

            print(
                f"P@{k}={metrics[f'precision@{k}']:.4f}  "
                f"R@{k}={metrics[f'recall@{k}']:.4f}  "
                f"NDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
                f"MRR={metrics['mrr']:.4f}"
            )

            # Accumulate for aggregate
            for metric_key, val in metrics.items():
                agg[system][metric_key] += val
            agg[system]["count"] += 1

            # Build detailed row — round here for clean CSV output
            row = {
                "query_id":          query_id,
                "query":             query,
                "system":            system,
                "gold_sections":     "|".join(gold_sections),
                "retrieved_headers": "|".join(retrieved_headers),
                "relevance_list":    str(relevance),
                "total_relevant":    total_relevant,
                **{k_: round(v, 4) for k_, v in metrics.items()},
                "gold_sections": "|".join(normalise(s) for s in gold_sections),
                "retrieved_header_chains": "|".join(
                    get_header_chain_for_logging(extract_metadata(r)) for r in results
                ),
            }
            per_query_rows.append(row)

    # Compute means
    aggregate_rows = []
    for system in RETRIEVAL_SYSTEMS:
        n = agg[system]["count"]
        if n == 0:
            continue
        aggregate_rows.append({
            "system":           system,
            f"mean_precision@{k}": round(agg[system][f"precision@{k}"] / n, 4),
            f"mean_recall@{k}":    round(agg[system][f"recall@{k}"] / n, 4),
            f"mean_ndcg@{k}":      round(agg[system][f"ndcg@{k}"] / n, 4),
            "mean_mrr":            round(agg[system]["mrr"] / n, 4),
            "map":                 round(agg[system]["ap"] / n, 4),
            "n_queries":           n,
        })

    return per_query_rows, aggregate_rows

# ── CSV output ────────────────────────────────────────────────────────────────

def save_results(per_query_rows: list[dict], aggregate_rows: list[dict]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Detailed per-query CSV
    detail_path = f"retrieval_eval_detail_{timestamp}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_query_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_query_rows)
    print(f"\nDetailed results saved to: {detail_path}")

    # Print summary table to console
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    header = f"{'System':<20} {'P@5':>6} {'R@5':>6} {'NDCG@5':>8} {'MRR':>6} {'MAP':>6}"
    print(header)
    print("-" * 60)
    for row in aggregate_rows:
        print(
            f"{row['system']:<20} "
            f"{row[f'mean_precision@{K}']:>6.4f} "
            f"{row[f'mean_recall@{K}']:>6.4f} "
            f"{row[f'mean_ndcg@{K}']:>8.4f} "
            f"{row['mean_mrr']:>6.4f} "
            f"{row['map']:>6.4f}"
        )
    print("=" * 60)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    per_query_rows, aggregate_rows = evaluate(EVAL_SET_PATH, k=K)
    save_results(per_query_rows, aggregate_rows)
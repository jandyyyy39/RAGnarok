from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config
from build_rules_rag import build_bm25_index

CHROMA_DIR = "data/chroma_db"

embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def test_retrieval_cosine_sim(query, k=5):
    print(f"\n--- Testing Query Naive Retrieval (Cosine Similarity): '{query}' ---")
    
    # Retrieve the top 5 most relevant chunks
    docs = vectorstore.similarity_search(query, k)
    
    if not docs:
        print("No results found. Did you build the database?")
        return

    for i, doc in enumerate(docs):
        print(f"\n[Result {i+1}]")
        # Print the metadata to prove the Markdown headers worked!
        print(f"Headers: {doc.metadata}")
        print(f"Content snippet: {doc.page_content[:200]}...\n")

    return docs

def test_bm25(bm25, final_splits, query, k=5):
    print(f"\n--- Testing Query Naive Retrieval (BM25): '{query}' ---")
    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = [final_splits[i] for i in top_k_indices]

    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"Score: {scores[top_k_indices[i]]}")
        print(f"Headers: {doc.metadata}")
        print(f"Content snippet: {doc.page_content[:200]}...\n")

    return results

def retrieve_with_rerank(query, candidate_k=20, final_k=5):
    print(f"\n--- Testing Two Stage Retrieval with Reranker: '{query}' ---")
    docs = vectorstore.similarity_search(query, k=candidate_k)

    pairs = [(query, doc.page_content) for doc in docs]

    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = [
        {
            "doc": doc,
            "text": doc.page_content,
            "metadata": doc.metadata,
            "rerank_score": float(score),
        }
        for doc, score in reranked[:final_k]
    ]

    for i, r in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"Rerank Score: {r['rerank_score']}")
        print(f"Headers: {r['metadata']}")
        print(f"Content snippet: {r['text'][:200]}...\n")

    return results

def retrieve_bm25_with_rerank(bm25, final_splits, query, candidate_k=20, final_k=5):
    print(f"\n--- Testing BM25 with Reranker: '{query}' ---")

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_candidate_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:candidate_k]

    candidate_docs = [final_splits[i] for i in top_candidate_indices]

    pairs = [(query, doc.page_content) for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidate_docs, rerank_scores, top_candidate_indices),
        key=lambda x: x[1],
        reverse=True
    )

    results = [
        {
            "doc": doc,
            "text": doc.page_content,
            "metadata": doc.metadata,
            "bm25_score": float(scores[idx]),
            "rerank_score": float(rerank_score),
        }
        for doc, rerank_score, idx in reranked[:final_k]
    ]

    for i, r in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"BM25 Score: {r['bm25_score']}")
        print(f"Rerank Score: {r['rerank_score']}")
        print(f"Headers: {r['metadata']}")
        print(f"Content snippet: {r['text'][:200]}...\n")

    return results

import re
from collections import Counter

def strip_anchor(text):
    return re.sub(r"\{#[^}]+\}", "", text).strip()

def get_deepest_header(metadata):
    for level in ("Header 3", "Header 2", "Header 1"):
        if level in metadata:
            return metadata[level]
    return ""

if __name__ == "__main__":
    # Cosine Sim
    test_retrieval_cosine_sim("I'm invisible and attacking from hiding. How does that work?")
    # test_retrieval_cosine_sim("What happens when I drop to 0 hit points?")

    # BM25
    bm25, final_splits = build_bm25_index()
    test_bm25(bm25, final_splits, "I'm invisible and attacking from hiding. How does that work?")
    # test_bm25(bm25, final_splits, "What happens when I drop to 0 hit points?")

    # Two Stage Retreival with Reranker (Cosine Sim)
    retrieve_with_rerank(query = "I'm invisible and attacking from hiding. How does that work?")
    # retrieve_with_rerank(query = "What happens when I drop to 0 hit points?")

    # Two Stage Retreival with Reranker (BM25)
    retrieve_bm25_with_rerank(bm25, final_splits, query = "I'm invisible and attacking from hiding. How does that work?")
    # retrieve_bm25_with_rerank(bm25, final_splits, query = "What happens when I drop to 0 hit points?")

    # canonical = sorted(set(
    # strip_anchor(get_deepest_header(chunk.metadata))
    # for chunk in final_splits
    # if get_deepest_header(chunk.metadata)
    # ))

    # for h in canonical:
    #     print(h)
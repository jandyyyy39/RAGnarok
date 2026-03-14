from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from build_rules_rag import build_bm25_index

CHROMA_DIR = "data/chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def test_retrieval_cosine_sim(query: str):
    print(f"\n--- Testing Query Naive Retrieval (Cosine Similarity): '{query}' ---")
    
    # Retrieve the top 5 most relevant chunks
    docs = vectorstore.similarity_search(query, k=5)
    
    if not docs:
        print("No results found. Did you build the database?")
        return

    for i, doc in enumerate(docs):
        print(f"\n[Result {i+1}]")
        # Print the metadata to prove the Markdown headers worked!
        print(f"Headers: {doc.metadata}")
        print(f"Content snippet: {doc.page_content[:200]}...\n")

def test_bm25(bm25, final_splits, query):
    print(f"\n--- Testing Query Naive Retrieval (BM25): '{query}' ---")
    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    results = [final_splits[i] for i in top_k_indices]

    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"Score: {scores[top_k_indices[i]]}")
        print(f"Headers: {doc.metadata}")
        print(f"Content snippet: {doc.page_content[:200]}...\n")

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

if __name__ == "__main__":
    # Cosine Sim
    test_retrieval_cosine_sim("How do I grapple someone?")
    test_retrieval_cosine_sim("What happens when I drop to 0 hit points?")

    # BM25
    bm25, final_splits = build_bm25_index()
    test_bm25(bm25, final_splits, "How do I grapple someone?")
    test_bm25(bm25, final_splits, "What happens when I drop to 0 hit points?")

    # Two Stage Retreival with Reranker
    retrieve_with_rerank(query = "How do I grapple someone?")
    retrieve_with_rerank(query = "What happens when I drop to 0 hit points?")
"""
Hybrid RAG Qualitative Trace Viewer
=====================================
Shows every step of the Hybrid (BM25 + Vector + Cross-Encoder) retrieval:
  - BM25 keyword search: tokens used, scores for top results, min-score filter
  - Vector similarity search: top-10 chunks
  - Merge + deduplication: which chunks came from BM25 only / Vector only / both
  - Cross-encoder re-ranking: score for every chunk in the merged pool
  - Final top-5 after re-ranking
  - The final ruling output

Usage:
    python evaluation/hybrid_trace.py -p local
    python evaluation/hybrid_trace.py -p groq
    python evaluation/hybrid_trace.py -p groq -q "I try to grapple the guard"
    python evaluation/hybrid_trace.py -p groq -i 2    # test case #2 from test_cases.json
"""

import sys
import os
import argparse
import time
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from groq import Groq

_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

DB_PATH = "data/chroma_db"
WORLD_CTX = "Location: The Black Boar Tavern. Active NPCs: Thrain Blackbeard."
MIN_BM25_SCORE = 1.0

STOP_WORDS = {
    "i", "me", "my", "we", "you", "he", "she", "it", "they", "them",
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for",
    "and", "or", "but", "if", "so", "as", "is", "am", "are", "was",
    "be", "been", "being", "do", "does", "did", "can", "will", "would",
    "could", "should", "may", "might", "shall", "have", "has", "had",
    "not", "no", "up", "out", "off", "with", "from", "into", "than",
    "that", "this", "these", "those", "what", "which", "who", "how",
    "want", "try", "get", "go", "let", "make", "take", "use", "see",
    "know", "think", "some", "more", "about", "there", "their", "then",
}


def divider(char="═", width=70):
    print(char * width)


class HybridTracer:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Tracer] Loading embeddings on {device.upper()}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': device})
        self.vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings)

        print(f"[Tracer] Loading BM25 index...")
        with open(Config.DATA_DIR / 'bm25_index.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(Config.DATA_DIR / 'chunks_raw.pkl', 'rb') as f:
            data = pickle.load(f)
            self.chunk_texts = data['texts']

        print(f"[Tracer] Loading cross-encoder reranker (once)...")
        self.ce_tokenizer = AutoTokenizer.from_pretrained(_RERANKER_MODEL)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(
            _RERANKER_MODEL, dtype=torch.float32).to(device)
        self.ce_model.eval()
        self.device = device
        print(f"[Tracer] Ready. ({len(self.chunk_texts)} BM25 chunks loaded)\n")

    def _get_ruling(self, chunks, player_action, world_context):
        context_text = "\n\n---\n\n".join(chunks)
        prompt = f"""SYSTEM: You are the Rules Arbiter for a D&D 5e game.
Your job is to interpret the player's intent based on the official rules.

WORLD CONTEXT: {world_context}
PLAYER ACTION: "{player_action}"

RELEVANT RULES FROM SRD:
{context_text}

INSTRUCTIONS:
1. If the action is trivial, state 'No check required' and explain briefly why.
2. If a check IS required, identify EXACTLY ONE primary Ability Check, Saving Throw, or Attack Roll.
3. For the resolution, use EXACTLY what the SRD rules above specify:
   - If the rules say it is a CONTESTED CHECK (one roll vs another), state both rolls (e.g. "Dexterity (Stealth) contested by Wisdom (Perception)"). Do NOT invent a DC.
   - If the rules specify a fixed DC or saving throw, state that DC (Easy=10, Medium=15, Hard=20, Very Hard=25).
4. Output strict, concise mechanics only. No conversational text."""

        resp = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise D&D 5e rules engine."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.1,
            max_tokens=150)
        return resp.choices[0].message.content

    def trace(self, player_action, world_context=WORLD_CTX):
        start_total = time.time()

        print()
        divider("═")
        print(f"  HYBRID TRACE — QUERY: \"{player_action}\"")
        divider("═")
        print(f"  Model   : {self.model}")
        print(f"  Context : {world_context}")
        print(f"  Strategy: BM25 keyword search (k=10) + Vector search (k=10)")
        print(f"            → Merge + deduplicate → Cross-encoder re-rank → top 5")
        divider("─")

        # ── BM25 KEYWORD SEARCH ───────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  STEP 1  |  BM25 Keyword Search  |  k=10  |  min_score={MIN_BM25_SCORE}")
        print(f"{'─'*70}\n")

        tokens = player_action.lower().split()
        tokenized = [t for t in tokens if t not in STOP_WORDS]
        removed = [t for t in tokens if t in STOP_WORDS]

        print(f"  Raw tokens    : {tokens}")
        print(f"  Stop words    : {removed}  ← removed (appear everywhere in SRD, no signal)")
        print(f"  Search tokens : {tokenized}\n")

        bm25_texts = []
        if not tokenized:
            print(f"  All tokens were stop words — BM25 skipped entirely.")
        else:
            bm25_scores = self.bm25.get_scores(tokenized)
            top_idx_all = np.argsort(bm25_scores)[::-1][:10]

            print(f"  BM25 top-10 results (score threshold = {MIN_BM25_SCORE}):\n")
            for rank, idx in enumerate(top_idx_all, 1):
                score = bm25_scores[idx]
                passed = score >= MIN_BM25_SCORE
                marker = "✓ KEPT" if passed else "✗ FILTERED (score too low)"
                print(f"  Rank {rank:2d}  BM25={score:.2f}  {marker}")
                print(f"         \"{self.chunk_texts[idx][:120].strip()}...\"")
                print()
                if passed:
                    bm25_texts.append(self.chunk_texts[idx])

        print(f"  → BM25 kept {len(bm25_texts)} chunk(s) above score threshold")

        # ── VECTOR SEARCH ─────────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  STEP 2  |  Vector Similarity Search  |  k=10")
        print(f"{'─'*70}\n")
        print(f"  Query: \"{player_action}\"\n")

        vector_results = self.vectorstore.similarity_search(player_action, k=10)
        vector_texts = [doc.page_content for doc in vector_results]

        for i, text in enumerate(vector_texts, 1):
            print(f"  Rank {i:2d}  (cosine similarity)")
            print(f"         \"{text[:120].strip()}...\"")
            print()

        print(f"  → Vector search returned {len(vector_texts)} chunk(s)")

        # ── MERGE + DEDUP ─────────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  STEP 3  |  Merge + Deduplicate")
        print(f"  (BM25 results first, then Vector — dedup by first 100 chars)")
        print(f"{'─'*70}\n")

        bm25_keys = {t[:100] for t in bm25_texts}
        vector_keys = {t[:100] for t in vector_texts}

        seen = set()
        merged = []
        sources = {}  # text_key -> source label

        for text in bm25_texts:
            key = text[:100]
            if key not in seen:
                seen.add(key)
                merged.append(text)
                sources[key] = "BM25 only"

        for text in vector_texts:
            key = text[:100]
            if key not in seen:
                seen.add(key)
                merged.append(text)
                sources[key] = "Vector only"
            elif key in bm25_keys:
                sources[key] = "BM25 + Vector"

        for i, text in enumerate(merged, 1):
            key = text[:100]
            src = sources.get(key, "unknown")
            print(f"  Chunk {i:2d}  [{src}]")
            print(f"         \"{text[:120].strip()}...\"")
            print()

        print(f"  → Merged pool: {len(merged)} unique chunk(s)")

        # ── CROSS-ENCODER RE-RANKING ──────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  STEP 4  |  Cross-Encoder Re-Ranking")
        print(f"  (Scores (query, chunk) pairs — much more accurate than cosine similarity)")
        print(f"  Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"{'─'*70}\n")

        raw_scores = []
        with torch.no_grad():
            for text in merged:
                enc = self.ce_tokenizer(
                    player_action, text,
                    truncation=True, max_length=512, return_tensors='pt'
                ).to(self.device)
                logit = self.ce_model(**enc).logits.view(-1)[0].item()
                raw_scores.append(logit)
        scores = np.array(raw_scores, dtype=np.float32)

        fallback_used = np.all(np.isnan(scores))
        if fallback_used:
            query_words = {w for w in player_action.lower().split() if w not in STOP_WORDS}
            source_bonus = np.array([
                1 if sources.get(text[:100]) == "BM25 + Vector" else 0 for text in merged
            ], dtype=np.float32)
            overlap = np.array([
                sum(1 for w in query_words if w in text.lower()) for text in merged
            ], dtype=np.float32)
            scores = overlap * 2 + source_bonus
            print(f"  ⚠  Cross-encoder returned NaN — falling back to word-overlap scoring")
            print(f"  Signal words: {sorted(query_words)}  (BM25+Vector chunks get +1 tiebreaker)\n")

        ranked_idx = np.argsort(scores)[::-1]
        score_label = "overlap" if fallback_used else "score"

        for rank, idx in enumerate(ranked_idx, 1):
            marker = "← TOP 5" if rank <= 5 else ""
            key = merged[idx][:100]
            src = sources.get(key, "unknown")
            print(f"  Rank {rank:2d}  {score_label}={scores[idx]:.4f}  [{src}]  {marker}")
            print(f"         \"{merged[idx][:120].strip()}...\"")
            print()

        # ── FINAL CHUNKS ─────────────────────────────────────────────────────
        final_idx = ranked_idx[:5]
        final_chunks = [merged[i] for i in final_idx]

        print(f"{'─'*70}")
        print(f"  FINAL CONTEXT  |  Top 5 after cross-encoder re-ranking")
        print(f"{'─'*70}\n")
        for i, text in enumerate(final_chunks, 1):
            print(f"  ── Chunk {i} ──────────────────────────────")
            print(f"  {text.strip()}")
            print()

        # ── FINAL RULING ─────────────────────────────────────────────────────
        print(f"{'─'*70}")
        print(f"  FINAL RULING")
        print(f"{'─'*70}\n")
        ruling = self._get_ruling(final_chunks, player_action, world_context)
        print(f"  {ruling.strip()}")

        elapsed = time.time() - start_total
        print(f"\n{'─'*70}")
        print(f"  Total time: {elapsed:.1f}s  |  LLM calls: 1 (ruling only — no LLM in retrieval)")
        divider("═")
        print()

        return {
            "query": player_action,
            "bm25_chunks": bm25_texts,
            "vector_chunks": vector_texts,
            "chunks_final": final_chunks,
            "ruling": ruling,
            "latency": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG qualitative trace viewer")
    parser.add_argument("-p", "--profile", choices=["local", "fast", "groq"], default="groq")
    parser.add_argument("-q", "--query", type=str, default=None)
    parser.add_argument("-i", "--id", type=int, default=None)
    args = parser.parse_args()

    if args.query:
        query = args.query
    elif args.id is not None:
        import json
        with open("evaluation/test_cases.json") as f:
            cases = json.load(f)
        match = next((c for c in cases if c["id"] == args.id), None)
        if not match:
            print(f"No test case with id={args.id}")
            sys.exit(1)
        query = match["query"]
        print(f"[Test case #{args.id}] category={match['category']}  type={match['query_type']}")
        print(f"[Expected mechanic] {match['description']}")
        print(f"[Primary keywords]  {match['primary_keywords']}")
    else:
        query = "I want to sneak past the sleeping guards"

    if args.profile == "local":
        print(f"Using LOCAL model: {Config.LLM_MODEL['LOCAL']}")
        client = OpenAI(base_url=Config.OLLAMA_BASE_URL, api_key="ollama")
        model = Config.LLM_MODEL['LOCAL']
    elif args.profile == "fast":
        print(f"Using GROQ FAST model: {Config.LLM_MODEL['GROQ_FAST']}")
        client = Groq(api_key=Config.GROQ_API_KEY)
        model = Config.LLM_MODEL['GROQ_FAST']
    else:
        print(f"Using GROQ model: {Config.LLM_MODEL['GROQ']}")
        client = Groq(api_key=Config.GROQ_API_KEY)
        model = Config.LLM_MODEL['GROQ']

    tracer = HybridTracer(client, model)
    tracer.trace(query)


if __name__ == "__main__":
    main()

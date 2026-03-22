"""
CRAG Qualitative Trace Viewer
==============================
Runs a single query through CRAG and prints every step verbosely:
  - All chunks retrieved in each round
  - The grade score for each chunk (1/2/3) + whether it was kept
  - The reformulated query (if Round 2 fires)
  - The final kept chunks passed to the LLM
  - The final ruling output

Usage:
    python evaluation/rag/crag_trace.py -p local
    python evaluation/rag/crag_trace.py -p groq
    python evaluation/rag/crag_trace.py -p groq -q "I try to grapple the guard"
    python evaluation/rag/crag_trace.py -p groq -i 2       # run test case #2 from test_cases.json
"""

import re
import sys
import os
import argparse
import time
import torch

# Make imports work regardless of execution directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)

from config import Config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from groq import Groq

DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
TEST_CASES_PATH = os.path.join(BASE_DIR, "evaluation", "rag", "test_cases.json")
WORLD_CTX = "Location: The Black Boar Tavern. Active NPCs: Thrain Blackbeard."

GRADE_LABELS = {1: "NOT RELEVANT", 2: "PARTIAL", 3: "DIRECT HIT"}
GRADE_SYMBOLS = {1: "✗", 2: "~", 3: "✓"}


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def divider(char="═", width=70):
    print(char * width)


def section(title):
    divider()
    print(f"  {title}")
    divider()


class CRAGTracer:
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
        print("[Tracer] Ready.\n")

    def _grade(self, doc_text, player_action):
        prompt = f"""You are a relevance grader for a D&D 5e rules database.
Score how relevant the rule excerpt is for determining the game mechanic
for the player action below.

Player action: "{player_action}"

Rule excerpt:
\"\"\"{doc_text[:500]}\"\"\"

Output ONLY a single digit: 1, 2, or 3.
1 = not relevant (different topic entirely)
2 = partially relevant (related mechanic but not exact)
3 = directly relevant (contains the specific rule needed)"""

        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.0,
            max_tokens=300)
        answer = _strip_think_tags(resp.choices[0].message.content).strip()
        for ch in answer:
            if ch in ('1', '2', '3'):
                return int(ch)
        return 1

    def _reformulate(self, player_action, world_context):
        prompt = f"""The following D&D player action did not return good results
from our rules database. Rewrite it as a precise D&D 5e rules lookup query.
Focus on the specific game mechanic, ability check, or spell involved.

Original action: "{player_action}"
Context: {world_context}

Output ONLY the rewritten query, nothing else."""

        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.1,
            max_tokens=300)
        return _strip_think_tags(resp.choices[0].message.content)

    def _get_ruling(self, relevant_docs, player_action, world_context):
        if not relevant_docs:
            return "No check required. No relevant D&D 5e rules apply to this action."

        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
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
        print(f"  CRAG TRACE — QUERY: \"{player_action}\"")
        divider("═")
        print(f"  Model   : {self.model}")
        print(f"  Context : {world_context}")
        divider("─")

        seen = set()
        scored = []  # (score, doc) tuples — sorted before final slice

        def _add_graded_trace(docs, threshold=2):
            added = 0
            for i, doc in enumerate(docs, 1):
                key = doc.page_content[:100]
                if key in seen:
                    print(f"  Chunk {i:2d}  [SKIPPED — already seen]")
                    continue
                seen.add(key)
                score = self._grade(doc.page_content, player_action)
                symbol = GRADE_SYMBOLS[score]
                label = GRADE_LABELS[score]
                print(f"  Chunk {i:2d}  Score={score} [{label}]  {symbol}")
                print(f"           \"{doc.page_content[:120].strip()}...\"")
                print()
                if score >= threshold:
                    scored.append((score, doc))
                    added += 1
            return added

        # ── ROUND 1 ──────────────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  ROUND 1  |  Vector similarity search  |  k=10")
        print(f"{'─'*70}")
        print(f"  Query used: \"{player_action}\"\n")

        docs1 = self.vectorstore.similarity_search(player_action, k=10)
        _add_graded_trace(docs1, threshold=2)
        print(f"  → Round 1 result: {len(scored)} relevant chunk(s) kept")

        # ── ROUND 2 ──────────────────────────────────────────────────────────
        reformulated_query = None
        if len(scored) < 3:
            print(f"\n{'─'*70}")
            print(f"  ROUND 2  |  Query reformulation + re-search  |  k=8")
            print(f"  (triggered because only {len(scored)} relevant chunk(s) found, need ≥ 3)")
            print(f"{'─'*70}")

            reformulated_query = self._reformulate(player_action, world_context)
            print(f"\n  Original query    : \"{player_action}\"")
            print(f"  Reformulated query: \"{reformulated_query}\"\n")

            docs2 = self.vectorstore.similarity_search(reformulated_query, k=8)
            r2_added = _add_graded_trace(docs2, threshold=2)
            print(f"  → Round 2 added {r2_added} more chunk(s)  |  Total relevant: {len(scored)}")
        else:
            print(f"  ✓ Enough relevant chunks found — Round 2 skipped")

        # ── ROUND 3 (fallback) ───────────────────────────────────────────────
        if len(scored) < 2:
            print(f"\n{'─'*70}")
            print(f"  ROUND 3  |  Emergency fallback  |  threshold=1 (accept anything loosely related)")
            print(f"  (triggered because still only {len(scored)} relevant chunk(s))")
            print(f"{'─'*70}\n")

            docs3 = self.vectorstore.similarity_search(player_action, k=5)
            r3_added = _add_graded_trace(docs3, threshold=1)
            print(f"  → Round 3 added {r3_added} more chunk(s)  |  Total relevant: {len(scored)}")
        else:
            print(f"  ✓ Round 3 fallback not needed")

        # Sort by score descending so Score=3 chunks appear first in final context
        scored.sort(key=lambda x: x[0], reverse=True)
        final_docs = [doc for _, doc in scored[:5]]

        print(f"\n  [Re-ranked by grade score — Score=3 chunks promoted to top]")

        # ── FINAL CHUNKS ─────────────────────────────────────────────────────
        final_chunks = final_docs
        print(f"\n{'─'*70}")
        print(f"  FINAL CONTEXT  |  {len(final_chunks)} chunk(s) passed to LLM")
        print(f"{'─'*70}\n")
        for i, doc in enumerate(final_chunks, 1):
            print(f"  ── Chunk {i} ──────────────────────────────")
            print(f"  {doc.page_content.strip()}")
            print()

        # ── FINAL RULING ─────────────────────────────────────────────────────
        print(f"{'─'*70}")
        print(f"  FINAL RULING")
        print(f"{'─'*70}\n")
        ruling = self._get_ruling(final_chunks, player_action, world_context)
        print(f"  {ruling.strip()}")

        elapsed = time.time() - start_total
        print(f"\n{'─'*70}")
        print(f"  Total time: {elapsed:.1f}s")
        divider("═")
        print()

        return {
            "query": player_action,
            "reformulated_query": reformulated_query,
            "rounds_used": 1 + (1 if reformulated_query else 0) + (1 if len(scored) < 2 else 0),
            "chunks_final": [d.page_content for d in final_chunks],
            "ruling": ruling,
            "latency": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="CRAG qualitative trace viewer")
    parser.add_argument("-p", "--profile", choices=["local", "fast", "groq"], default="groq",
                        help="Model profile (local=Ollama, groq=Groq, fast=Groq fast)")
    parser.add_argument("-q", "--query", type=str, default=None,
                        help="Custom query string to trace")
    parser.add_argument("-i", "--id", type=int, default=None,
                        help="Test case ID from evaluation/rag/test_cases.json (1-50)")
    args = parser.parse_args()

    # --- Select query ---
    if args.query:
        query = args.query
    elif args.id is not None:
        import json
        with open(TEST_CASES_PATH) as f:
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
        # Default: a good example for slides
        query = "I want to sneak past the sleeping guards"

    # --- Init client ---
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

    tracer = CRAGTracer(client, model)
    tracer.trace(query)


if __name__ == "__main__":
    main()

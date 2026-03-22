"""
Naive RAG Qualitative Trace Viewer
=====================================
Shows exactly what the baseline Naive RAG does step-by-step:
  - The raw query sent to vector search
  - All k=5 chunks returned (ranked by cosine similarity)
  - No grading, no reformulation — what you get is what you pass to the LLM
  - The final ruling output

Usage:
    python evaluation/naive_trace.py -p local
    python evaluation/naive_trace.py -p groq
    python evaluation/naive_trace.py -p groq -q "I try to grapple the guard"
    python evaluation/naive_trace.py -p groq -i 2    # test case #2 from test_cases.json
"""

import sys
import os
import argparse
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from groq import Groq

DB_PATH = "data/chroma_db"
WORLD_CTX = "Location: The Black Boar Tavern. Active NPCs: Thrain Blackbeard."


def divider(char="═", width=70):
    print(char * width)


class NaiveTracer:
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
        print(f"  NAIVE TRACE — QUERY: \"{player_action}\"")
        divider("═")
        print(f"  Model   : {self.model}")
        print(f"  Context : {world_context}")
        divider("─")

        # ── RETRIEVAL ────────────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  RETRIEVAL  |  Direct vector similarity search  |  k=5")
        print(f"  (No grading. No reformulation. Top-5 cosine similarity, passed straight to LLM.)")
        print(f"{'─'*70}")
        print(f"  Query: \"{player_action}\"\n")

        docs = self.vectorstore.similarity_search(player_action, k=5)
        chunks = [doc.page_content for doc in docs]

        for i, chunk in enumerate(chunks, 1):
            print(f"  ── Rank {i} (cosine similarity) ──────────────────────────")
            print(f"  {chunk.strip()}")
            print()

        # ── FINAL RULING ─────────────────────────────────────────────────────
        print(f"{'─'*70}")
        print(f"  FINAL RULING  (all {len(chunks)} chunks passed directly to LLM)")
        print(f"{'─'*70}\n")
        ruling = self._get_ruling(chunks, player_action, world_context)
        print(f"  {ruling.strip()}")

        elapsed = time.time() - start_total
        print(f"\n{'─'*70}")
        print(f"  Total time: {elapsed:.1f}s")
        divider("═")
        print()

        return {
            "query": player_action,
            "chunks_final": chunks,
            "ruling": ruling,
            "latency": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="Naive RAG qualitative trace viewer")
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

    tracer = NaiveTracer(client, model)
    tracer.trace(query)


if __name__ == "__main__":
    main()

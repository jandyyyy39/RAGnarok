import re
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks produced by deepseek-r1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class RulesArbiterCRAG:
    def __init__(self, client, model_profile: str, db_path="chroma_db"):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[CRAG Arbiter] Using {device.upper()} for embeddings.")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': device})
        

        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings)

    def _grade_document(self, doc_text, player_action):
        """
        Score relevance 1-3 instead of binary yes/no.
        1 = not relevant, 2 = partially relevant, 3 = directly relevant.
        Threshold >= 2 keeps more useful chunks that a hard yes/no would drop.
        """
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
        # Extract first digit found; default to 1 (not relevant) if parse fails
        for ch in answer:
            if ch in ('1', '2', '3'):
                return int(ch)
        return 1

    def _reformulate_query(self, player_action, world_context):
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

    def retrieve(self, player_action, world_context=""):
        seen = set()
        scored = []  # list of (score, doc) so we can sort by score before slicing

        def _add_graded(docs, threshold=2):
            """Grade docs and append (score, doc) tuples for those scoring >= threshold."""
            for doc in docs:
                key = doc.page_content[:100]
                if key in seen:
                    continue
                seen.add(key)
                score = self._grade_document(doc.page_content, player_action)
                if score >= threshold:
                    scored.append((score, doc))

        # Round 1: wider initial pool (k=10 instead of 6) — more candidates to grade
        docs1 = self.vectorstore.similarity_search(player_action, k=10)
        _add_graded(docs1, threshold=2)

        # Round 2: reformulate and retry if fewer than 3 relevant found
        if len(scored) < 3:
            new_query = self._reformulate_query(player_action, world_context)
            docs2 = self.vectorstore.similarity_search(new_query, k=8)
            _add_graded(docs2, threshold=2)

        # Round 3 (keyword safety net): if still weak, fall back to direct query
        # with a looser threshold so we don't return empty-handed
        if len(scored) < 2:
            docs3 = self.vectorstore.similarity_search(player_action, k=5)
            _add_graded(docs3, threshold=1)  # accept anything loosely related

        # Sort by score descending so Score=3 chunks always appear first in the
        # final context window, regardless of their cosine similarity rank.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:5]]

    def get_ruling(self, player_action, world_context):
        relevant_rules = self.retrieve(player_action, world_context)

        if not relevant_rules:
            return {
                "ruling"        : "No check required. No relevant D&D 5e rules apply to this action.",
                "usage"         : 0,
                "context_text"  : "",
            }

        context_text = "\n".join([doc.page_content for doc in relevant_rules])

        prompt = f"""SYSTEM: You are the Rules Arbiter for a D&D 5e game.
            Your job is to interpret the player's intent based on the official rules.

            WORLD CONTEXT: {world_context}
            PLAYER ACTION: "{player_action}"

            RELEVANT RULES FROM SRD:
            {context_text}

            INSTRUCTIONS:
            1. If the action is trivial (e.g., greeting someone, walking, eating, sitting down, accepting a quest), state 'No check required' and explain briefly why.
            2. If a check IS required, identify EXACTLY ONE primary Ability Check, Saving Throw, or Attack Roll.
            3. For the resolution, use EXACTLY what the SRD rules above specify:
               - If the rules say it is a CONTESTED CHECK (one roll vs another), state both rolls (e.g. "Dexterity (Stealth) contested by Wisdom (Perception)"). Do NOT invent a DC.
               - If the rules specify a fixed DC or saving throw, state that DC (Easy=10, Medium=15, Hard=20, Very Hard=25).
            4. Output strict, concise mechanics only. No conversational text."""

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise D&D 5e rules engine."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.1,
            max_tokens=150)
        
        total_tokens = response.usage.total_tokens if getattr(response, 'usage', None) else 0

        return {
            "ruling"        : response.choices[0].message.content,
            "usage"         : total_tokens,
            "context_text"  : context_text,
        }
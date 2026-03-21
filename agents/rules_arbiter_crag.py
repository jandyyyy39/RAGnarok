import re
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class RulesArbiterCRAG:
    _INITIAL_K = 10
    _REFORMULATE_K = 8
    _FALLBACK_K = 5
    _FINAL_TOP_K = 5
    _MAIN_THRESHOLD = 2
    _FALLBACK_THRESHOLD = 1
    _MIN_RELEVANT_BEFORE_REWRITE = 3
    _MIN_RELEVANT_BEFORE_FALLBACK = 2

    def __init__(self, client, model_profile: str, db_path="chroma_db"):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CRAG Arbiter] Using {self.device.upper()} for embeddings.")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": self.device},
        )

        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )

    @staticmethod
    def _doc_text(doc) -> str:
        return getattr(doc, "page_content", "") or ""

    @staticmethod
    def _doc_metadata(doc) -> dict:
        return getattr(doc, "metadata", {}) or {}

    def _doc_key(self, doc) -> str:
        return self._doc_text(doc)

    def _grade_document(self, doc_text, player_action):
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
            max_tokens=300,
        )
        answer = _strip_think_tags(resp.choices[0].message.content).strip()

        for ch in answer:
            if ch in ("1", "2", "3"):
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
            max_tokens=300,
        )
        return _strip_think_tags(resp.choices[0].message.content)

    def retrieve(self, player_action, world_context=""):
        seen = set()
        scored = []

        def _add_graded(docs, threshold=2):
            for doc in docs:
                key = self._doc_key(doc)
                if not key or key in seen:
                    continue

                seen.add(key)
                score = self._grade_document(self._doc_text(doc), player_action)
                if score >= threshold:
                    scored.append((score, doc))

        docs1 = self.vectorstore.similarity_search(player_action, k=self._INITIAL_K)
        _add_graded(docs1, threshold=self._MAIN_THRESHOLD)

        if len(scored) < self._MIN_RELEVANT_BEFORE_REWRITE:
            new_query = self._reformulate_query(player_action, world_context)
            docs2 = self.vectorstore.similarity_search(new_query, k=self._REFORMULATE_K)
            _add_graded(docs2, threshold=self._MAIN_THRESHOLD)

        if len(scored) < self._MIN_RELEVANT_BEFORE_FALLBACK:
            docs3 = self.vectorstore.similarity_search(player_action, k=self._FALLBACK_K)
            _add_graded(docs3, threshold=self._FALLBACK_THRESHOLD)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:self._FINAL_TOP_K]]

    def get_ruling(self, player_action, world_context):
        relevant_rules = self.retrieve(player_action, world_context)

        if not relevant_rules:
            return {
                "ruling": "No check required. No relevant D&D 5e rules apply to this action.",
                "usage": 0,
                "context_text": "",
            }

        context_text = "\n".join(self._doc_text(doc) for doc in relevant_rules)

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
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
            max_tokens=150,
        )

        total_tokens = response.usage.total_tokens if getattr(response, "usage", None) else 0

        return {
            "ruling": response.choices[0].message.content,
            "usage": total_tokens,
            "context_text": context_text,
        }
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
        prompt = f"""You are a relevance grader for a D&D 5e rules database.
            Does the following rule excerpt contain information DIRECTLY relevant
            to determining the game mechanic for this player action?

            Player action: "{player_action}"

            Rule excerpt:
            \"\"\"{doc_text[:500]}\"\"\"

            Answer ONLY 'yes' or 'no'. Nothing else."""

        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.0,
            max_tokens=300)   # deepseek-r1 needs room for <think>...</think> before outputting yes/no
        answer = _strip_think_tags(resp.choices[0].message.content)
        return 'yes' in answer.lower()

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
            max_tokens=300)   # deepseek-r1 needs room for <think>...</think> before the rewritten query
        
        return _strip_think_tags(resp.choices[0].message.content)

    def retrieve(self, player_action, world_context=""):
        # Round 1: retrieve and grade
        docs = self.vectorstore.similarity_search(player_action, k=6)

        relevant = []
        for doc in docs:
            if self._grade_document(doc.page_content, player_action):
                relevant.append(doc)

        # Round 2: if too few relevant, reformulate and retry
        if len(relevant) < 2:
            new_query = self._reformulate_query(player_action, world_context)
            docs2 = self.vectorstore.similarity_search(new_query, k=6)

            existing_texts = {r.page_content for r in relevant}
            for doc in docs2:
                if doc.page_content not in existing_texts:
                    if self._grade_document(doc.page_content, player_action):
                        relevant.append(doc)
                        existing_texts.add(doc.page_content)

        return relevant[:5]

    def get_ruling(self, player_action, world_context):
        relevant_rules = self.retrieve(player_action, world_context)

        if not relevant_rules:
            return "No check required. No relevant D&D 5e rules apply to this action."

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
            3. Set a single Difficulty Class (DC) using the standard table (Easy=10, Medium=15, Hard=20, Very Hard=25).
            4. Output strict, concise mechanics only. No conversational text."""

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise D&D 5e rules engine."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.1,
            max_tokens=150)
        return response.choices[0].message.content
import re
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks produced by deepseek-r1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



class RulesArbiterHyDE:
    # Init is similar to the BL Naive arbiter. Only diff is how it searches.
    def __init__(self, client, model_profile: str, db_path="chroma_db"):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[HyDE Arbiter] Using {device.upper()} for embeddings.")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': device})
        
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings)

    def _generate_hypothetical_document(self, player_action, world_context):
        prompt = f"""You are a D&D 5e rules encyclopedia.
        Write a single paragraph (3-5 sentences) from the official D&D 5e SRD
        that contains the EXACT rules mechanic for this player action.

        Use precise SRD language: skill names (Athletics, Acrobatics, Stealth,
        Deception, Persuasion, Perception, Animal Handling, Medicine, etc.),
        action types (action, bonus action, reaction), and mechanical terms
        (DC, contested check, saving throw, ability check).

        Player action: "{player_action}"
        Game context: {world_context}

        Begin directly with the mechanic name or rule heading. Write ONLY the
        rules text as it would appear in the SRD. No commentary.
        If no ability check or roll is needed, write:
        'This action does not require an ability check or saving throw. It succeeds automatically.'"""

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.1,
            max_tokens=600)   # deepseek-r1 needs ~300-400 tokens just for <think>...</think> before writing the answer
        return _strip_think_tags(response.choices[0].message.content)

    def retrieve(self, player_action, world_context=""):
        # Generate 3 hypothetical documents to reduce variance from small models.
        # Different phrasings catch different chunks.
        hyp_docs = [
            self._generate_hypothetical_document(player_action, world_context)
            for _ in range(3)
        ]

        seen = set()
        merged = []

        # Search with each hypothetical doc (k=8 each for broader recall)
        for hyp in hyp_docs:
            for doc in self.vectorstore.similarity_search(hyp, k=8):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        # Also search with the original player query directly.
        # Small models sometimes produce hypothetical docs that miss obvious
        # SRD terms that the raw query would have matched (e.g. Q29 acid splash).
        for doc in self.vectorstore.similarity_search(player_action, k=5):
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        # Re-rank the merged pool by how many query words appear in each chunk,
        # then return the top 5. This lightweight filter surfaces chunks that
        # actually share vocabulary with the player's action.
        query_words = set(player_action.lower().split())
        def overlap_score(doc):
            return sum(1 for w in query_words if w in doc.page_content.lower())

        merged.sort(key=overlap_score, reverse=True)
        return merged[:5]



    def get_ruling(self, player_action, world_context):
        relevant_rules = self.retrieve(player_action, world_context)
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
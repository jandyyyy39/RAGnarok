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

    # Stop words that appear everywhere in the SRD and carry no retrieval signal.
    # Filtering these before overlap scoring prevents common English words from
    # inflating scores for irrelevant chunks.
    _STOP_WORDS = {
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

    # Focus angles for each hypothesis — forces the model to approach
    # the mechanic from a different angle each time, preventing identical outputs.
    _HYPOTHESIS_FOCUS = [
        "Focus on the PRIMARY ability check or attack roll and how it is resolved.",
        "Focus on the CONDITIONS, REQUIREMENTS, or LIMITATIONS that apply to this action.",
        "Focus on the CONSEQUENCES, EFFECTS, or what happens after the roll succeeds or fails.",
    ]

    def _generate_hypothetical_document(self, player_action, world_context, focus: str = "", temperature: float = 0.1):
        focus_line = f"\n        FOCUS: {focus}" if focus else ""
        prompt = f"""You are a D&D 5e rules encyclopedia.
        Write a single paragraph (3-5 sentences) from the official D&D 5e SRD
        that contains the EXACT rules mechanic for this player action.

        Use precise SRD language: skill names (Athletics, Acrobatics, Stealth,
        Deception, Persuasion, Perception, Animal Handling, Medicine, etc.),
        action types (action, bonus action, reaction), and mechanical terms
        (DC, contested check, saving throw, ability check).{focus_line}

        Player action: "{player_action}"
        Game context: {world_context}

        Begin directly with the mechanic name or rule heading. Write ONLY the
        rules text as it would appear in the SRD. No commentary.
        If no ability check or roll is needed, write:
        'This action does not require an ability check or saving throw. It succeeds automatically.'"""

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=temperature,
            max_tokens=600)
        return _strip_think_tags(response.choices[0].message.content)

    @staticmethod
    def _normalize_key(text: str) -> str:
        """Strip markdown links and normalise for near-duplicate detection.
        e.g. '[grappled](#grappled)' and 'grappled' map to the same key."""
        stripped = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        return stripped[:150].lower().strip()

    def retrieve(self, player_action, world_context=""):
        # Generate 3 hypothetical documents with different focus angles and
        # escalating temperature so each hypothesis explores a different aspect.
        hyp_docs = [
            self._generate_hypothetical_document(
                player_action, world_context,
                focus=self._HYPOTHESIS_FOCUS[i],
                temperature=0.1 + i * 0.2  # 0.1, 0.3, 0.5
            )
            for i in range(3)
        ]

        seen = set()
        merged = []

        def _add(doc):
            key = self._normalize_key(doc.page_content)
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        # Search with each hypothetical doc (k=8 each for broader recall)
        for hyp in hyp_docs:
            for doc in self.vectorstore.similarity_search(hyp, k=8):
                _add(doc)

        # Also search with the original player query directly.
        # Small models sometimes produce hypothetical docs that miss obvious
        # SRD terms that the raw query would have matched (e.g. Q29 acid splash).
        for doc in self.vectorstore.similarity_search(player_action, k=5):
            _add(doc)

        # Re-rank by query word overlap, filtering stop words so common English
        # words don't inflate scores for irrelevant chunks.
        query_words = {w for w in player_action.lower().split() if w not in self._STOP_WORDS}

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
    
        return response.choices[0].message.content        
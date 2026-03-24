import re
import torch
from concurrent.futures import ThreadPoolExecutor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config
from time import time, sleep


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------

_RULES_HYPOTHESIS_FOCUS = [
    "Focus on the PRIMARY ability check or attack roll and how it is resolved.",
    "Focus on the CONDITIONS, REQUIREMENTS, or LIMITATIONS that apply to this action.",
    "Focus on the CONSEQUENCES, EFFECTS, or what happens after the roll succeeds or fails.",
]

# Derived directly from EXPLORATION_CONTEXT vocabulary written by the original author.
_EXPLORATION_HYPOTHESIS_FOCUS = [
    (
        "Focus on terrain, movement, and travel rules: difficult terrain, travel pace, "
        "climbing, swimming, crawling, jumping, falling, mounts, vehicles, forced march, "
        "navigating, getting lost, tracking, foraging, and marching order."
    ),
    (
        "Focus on vision, light, cover, and stealth rules: darkvision, blindsight, "
        "truesight, dim light, bright light, heavily obscured, lightly obscured, "
        "half cover, three-quarters cover, hiding, invisible, unseen, searching, "
        "secret doors, and Perception or Investigation checks."
    ),
    (
        "Focus on hazards, survival limits, and object interaction: exhaustion, "
        "suffocation, starvation, dehydration, extreme heat, extreme cold, strong wind, "
        "heavy precipitation, high altitude, encumbrance, carrying capacity, resting, "
        "short rest, long rest, hit dice, objects AC and HP, breaking, traps, "
        "locks, thieves tools, triggers, and disarming mechanisms."
    ),
]

_RULES_SYSTEM_FRAMING = "You are a D&D 5e rules encyclopedia."
_EXPLORATION_SYSTEM_FRAMING = "You are a D&D 5e exploration and environment rules encyclopedia."

_RULES_INSTRUCTION = (
    "Write a single paragraph (3-5 sentences) from the official D&D 5e SRD "
    "that contains the EXACT rules mechanic for this player action.\n\n"
    "Use precise SRD language: skill names (Athletics, Acrobatics, Stealth, "
    "Deception, Persuasion, Perception, Animal Handling, Medicine, etc.), "
    "action types (action, bonus action, reaction), and mechanical terms "
    "(DC, contested check, saving throw, ability check)."
)

_EXPLORATION_INSTRUCTION = (
    "Write a single paragraph (3-5 sentences) from the official D&D 5e SRD "
    "that contains the EXACT rules mechanic for this player action.\n\n"
    "Use precise SRD language for exploration: terrain types, travel pace, "
    "vision categories (darkvision, blindsight, truesight), cover types "
    "(half cover, three-quarters cover), environmental hazards (exhaustion, "
    "suffocation, extreme heat, extreme cold), survival checks (Survival, "
    "Perception, Investigation), object AC and HP, and trap/lock mechanics."
)


class HyDERetriever:
    """
    Unified Hypothetical Document Embedding retriever for both rules_logic
    and world_exploration skills. Both modes query the same ChromaDB
    collection; mode controls hypothesis framing only.

    Usage:
        hyde_rules = HyDERetriever(client, model_profile, mode="rules")
        hyde_exploration = HyDERetriever(client, model_profile, mode="exploration")

        docs, tokens = hyde_rules.retrieve(player_action, world_context)
        docs, tokens = hyde_exploration.retrieve(player_action, world_context)
    """

    # Class-level singletons — shared across both instances to avoid loading
    # the embedding model twice.
    _embeddings = None
    _vectorstore = None

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

    _HYP_K = 8
    _DIRECT_K = 5
    _FINAL_TOP_K = 5

    def __init__(self, client, model_profile: str, mode: str = "rules", hyde_model: str = None):
        assert mode in ("rules", "exploration"), f"Invalid mode: {mode}"
        self.client = client
    
        if not hyde_model:
            self.model = Config.LLM_MODEL.get("LOCAL_HYDE" if model_profile == "LOCAL" else "GROQ_HYDE") \
                or Config.LLM_MODEL.get("LOCAL_FAST" if model_profile == "LOCAL" else "GROQ_FAST")
        else:
            self.model = hyde_model
        
        self.mode = mode

        if mode == "rules":
            self._hypothesis_focus = _RULES_HYPOTHESIS_FOCUS
            self._system_framing = _RULES_SYSTEM_FRAMING
            self._instruction = _RULES_INSTRUCTION
        else:
            self._hypothesis_focus = _EXPLORATION_HYPOTHESIS_FOCUS
            self._system_framing = _EXPLORATION_SYSTEM_FRAMING
            self._instruction = _EXPLORATION_INSTRUCTION

        # Initialise shared vectorstore once across all instances
        HyDERetriever._get_vectorstore()
        print(f"[HyDERetriever] Initialized in '{mode}' mode.")

    @classmethod
    def _get_vectorstore(cls):
        if cls._vectorstore is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[HyDERetriever] Loading embedding model on {device.upper()}...")
            cls._embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": device},
            )
            cls._vectorstore = Chroma(
                persist_directory=str(Config.DATA_DIR / "chroma_db"),
                embedding_function=cls._embeddings,
            )
        return cls._vectorstore

    # ------------------------------------------------------------------
    # Internal helpers (unchanged from original RulesArbiterHyDE)
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_text(doc) -> str:
        return getattr(doc, "page_content", "") or ""

    @staticmethod
    def _doc_metadata(doc) -> dict:
        return getattr(doc, "metadata", {}) or {}

    @staticmethod
    def _normalize_key(text: str) -> str:
        stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        return stripped.lower().strip()

    def _doc_key(self, doc) -> str:
        return self._normalize_key(self._doc_text(doc))

    # ------------------------------------------------------------------
    # Hypothesis generation — now returns (text, tokens)
    # ------------------------------------------------------------------

    def _generate_hypothetical_document(
        self,
        player_action: str,
        world_context: str,
        focus: str = "",
        temperature: float = 0.1,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> tuple[str, int]:
        focus_line = f"\n        FOCUS: {focus}" if focus else ""
        prompt = (
            f"{self._system_framing}\n"
            f"{self._instruction}{focus_line}\n\n"
            f"Player action: \"{player_action}\"\n"
            f"Game context: {world_context}\n\n"
            "Begin directly with the mechanic name or rule heading. Write ONLY the "
            "rules text as it would appear in the SRD. No commentary.\n"
            "If no ability check or roll is needed, write:\n"
            "'This action does not require an ability check or saving throw. "
            "It succeeds automatically.'"
        )

        last_exc = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=temperature,
                    max_tokens=600,
                )
                tokens = response.usage.total_tokens if getattr(response, "usage", None) else 0
                return _strip_think_tags(response.choices[0].message.content), tokens

            except Exception as e:
                last_exc = e
                print(f"[HyDE:{self.mode}] Attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    sleep(retry_delay)
            tokens = response.usage.total_tokens if getattr(response, "usage", None) else 0

        print(f"[HyDE:{self.mode}] All retries exhausted. Last error: {last_exc}. Returning empty hypothesis.")
        return "", 0

    # ------------------------------------------------------------------
    # Public retrieval — returns (docs, total_hyde_tokens)
    # ------------------------------------------------------------------

    def retrieve(self, player_action: str, world_context: str = "") -> tuple[list, int]:
        hyde_tokens = 0
        tag = f"[HyDE:{self.mode}]"
        print(f"{tag} Starting retrieval for: '{player_action[:60]}'")
        print(f"{tag} Generating 3 hypothetical documents in parallel...")

        t_hyp = time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    self._generate_hypothetical_document,
                    player_action,
                    world_context,
                    self._hypothesis_focus[i],
                    0.1 + i * 0.2,
                )
                for i in range(3)
            ]
            results = [f.result() for f in futures]
        hyp_elapsed = round((time() - t_hyp) * 1000)

        hyp_docs = []
        for i, (text, tokens) in enumerate(results):
            hyp_docs.append(text)
            hyde_tokens += tokens
            print(f"{tag} Hypothesis {i+1}: {tokens} tokens | '{text}'")
        print(f"{tag} Hypothesis generation complete: {hyde_tokens} tokens total | {hyp_elapsed}ms")

        vs = self._get_vectorstore()
        seen = set()
        merged = []

        def _add(doc):
            key = self._doc_key(doc)
            if not key or key in seen:
                return
            seen.add(key)
            merged.append(doc)

        t_search = time()
        for hyp in hyp_docs:
            for doc in vs.similarity_search(hyp, k=self._HYP_K):
                _add(doc)

        pre_direct = len(merged)
        for doc in vs.similarity_search(player_action, k=self._DIRECT_K):
            _add(doc)
        search_elapsed = round((time() - t_search) * 1000)

        print(f"{tag} Vector search: {pre_direct} docs from hypotheses + {len(merged) - pre_direct} from direct | {len(merged)} unique | {search_elapsed}ms")

        query_words = {
            w for w in player_action.lower().split()
            if w not in self._STOP_WORDS
        }

        def overlap_score(doc):
            text = self._doc_text(doc).lower()
            return sum(1 for w in query_words if w in text)

        merged.sort(key=overlap_score, reverse=True)
        final = merged[:self._FINAL_TOP_K]
        print(f"{tag} Final: {len(final)} docs returned to orchestrator | {hyde_tokens} total tokens")
        return final, hyde_tokens
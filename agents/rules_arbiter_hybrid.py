import torch
import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RulesArbiterHybrid:
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

    _MIN_BM25_SCORE = 1.0
    _BM25_TOP_K = 10
    _VECTOR_TOP_K = 10
    _FINAL_TOP_K = 5

    def __init__(self, client, model_profile: str, db_path="chroma_db"):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Hybrid Arbiter] Using {self.device.upper()} for embeddings.")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": self.device},
        )

        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )

        with open(Config.DATA_DIR / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(Config.DATA_DIR / "bm25_chunks.pkl", "rb") as f:
            self.final_splits = pickle.load(f)

        self.chunk_texts = [doc.page_content for doc in self.final_splits]

        self.ce_tokenizer = AutoTokenizer.from_pretrained(_RERANKER_MODEL)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(
            _RERANKER_MODEL,
            dtype=torch.float32,
        ).to(self.device)
        self.ce_model.eval()

        print(
            f"[Hybrid Arbiter] BM25 ({len(self.chunk_texts)} chunks) + "
            f"Vector + Reranker ready."
        )

    @staticmethod
    def _doc_text(doc) -> str:
        return getattr(doc, "page_content", "") or ""

    @staticmethod
    def _doc_metadata(doc) -> dict:
        return getattr(doc, "metadata", {}) or {}

    def _doc_key(self, doc) -> str:
        return self._doc_text(doc)

    def _ce_scores(self, query: str, docs: list[str]) -> np.ndarray:
        scores = []
        with torch.no_grad():
            for doc in docs:
                enc = self.ce_tokenizer(
                    query,
                    doc,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logit = self.ce_model(**enc).logits.view(-1)[0].item()
                scores.append(logit)

        scores = np.array(scores, dtype=np.float32)

        if np.all(np.isnan(scores)):
            query_words = {
                w for w in query.lower().split()
                if w not in self._STOP_WORDS
            }
            scores = np.array(
                [sum(1 for w in query_words if w in doc.lower()) for doc in docs],
                dtype=np.float32,
            )

        return scores

    def _bm25_retrieve(self, player_action: str, top_k: int = None):
        top_k = top_k or self._BM25_TOP_K

        tokens = player_action.lower().split()
        tokenized_query = [t for t in tokens if t not in self._STOP_WORDS]

        if not tokenized_query:
            return []

        bm25_scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(bm25_scores)[::-1]

        kept_indices = [
            i for i in ranked_indices[:top_k]
            if bm25_scores[i] >= self._MIN_BM25_SCORE
        ]

        return [self.final_splits[i] for i in kept_indices]

    def _vector_retrieve(self, player_action: str, top_k: int = None):
        top_k = top_k or self._VECTOR_TOP_K
        return self.vectorstore.similarity_search(player_action, k=top_k)

    def _merge_dedupe_docs(self, docs: list):
        seen = set()
        merged = []

        for doc in docs:
            key = self._doc_key(doc)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(doc)

        return merged

    def retrieve(self, player_action, world_context=""):
        bm25_docs = self._bm25_retrieve(player_action, top_k=self._BM25_TOP_K)
        vector_docs = self._vector_retrieve(player_action, top_k=self._VECTOR_TOP_K)

        merged_docs = self._merge_dedupe_docs(bm25_docs + vector_docs)

        if not merged_docs:
            return []

        doc_texts = [self._doc_text(doc) for doc in merged_docs]
        scores = self._ce_scores(player_action, doc_texts)
        ranked_idx = np.argsort(scores)[::-1][:self._FINAL_TOP_K]

        return [merged_docs[i] for i in ranked_idx]

    def get_ruling(self, player_action, world_context):
        relevant_rules = self.retrieve(player_action, world_context)
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
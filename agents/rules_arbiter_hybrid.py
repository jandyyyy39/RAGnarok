import torch
import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from config import Config


class RulesArbiterHybrid:
    def __init__(self, client, model_profile: str, db_path="chroma_db"):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Hybrid Arbiter] Using {device.upper()} for embeddings.")


        # Vector store (same as Naive)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': device})
        

        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings)
        

        # BM25 keyword index (loaded from the pickle you built in Step 4)
        with open(Config.DATA_DIR / 'bm25_index.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(Config.DATA_DIR / 'chunks_raw.pkl', 'rb') as f:
            data = pickle.load(f)
            self.chunk_texts = data['texts']

        # Cross-encoder re-ranker
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device=device)
        print(f"[Hybrid Arbiter] BM25 ({len(self.chunk_texts)} chunks) + Vector + Reranker ready.")

    # Common words that appear everywhere in the SRD and carry no signal.
    # Leaving these in causes BM25 to rank random character background tables
    # (full of "I believe...", "I try to...") above actual rule chunks.
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

    # Minimum BM25 score for a result to enter the merged pool.
    # Results below this threshold are noise (no meaningful keyword overlap).
    _MIN_BM25_SCORE = 1.0

    def retrieve(self, player_action, world_context=""):
        # 1. BM25 keyword search — top 10 (after stop-word filtering)
        tokens = player_action.lower().split()
        tokenized_query = [t for t in tokens if t not in self._STOP_WORDS]

        if not tokenized_query:
            # Query was all stop words — skip BM25 entirely
            bm25_docs = []
        else:
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # Only keep results that actually matched something meaningful
            bm25_top_idx = [
                i for i in np.argsort(bm25_scores)[::-1][:10]
                if bm25_scores[i] >= self._MIN_BM25_SCORE
            ]
            bm25_docs = [self.chunk_texts[i] for i in bm25_top_idx]

        # 2. Vector similarity search — top 10
        vector_results = self.vectorstore.similarity_search(player_action, k=10)
        vector_docs = [doc.page_content for doc in vector_results]

        # 3. Merge and de dupe
        seen = set()
        merged = []
        for doc_text in bm25_docs + vector_docs:
            key = doc_text[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc_text)

        if not merged:
            return []

        # 4. Re-rank with cross-encoder
        pairs = [(player_action, doc) for doc in merged]
        scores = self.reranker.predict(pairs)
        ranked_idx = np.argsort(scores)[::-1][:5]

        # Wrap results so they have .page_content like LangChain docs
        class DocWrapper:
            def __init__(self, text):
                self.page_content = text
                self.metadata = {}

        return [DocWrapper(merged[i]) for i in ranked_idx]

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
                {"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.1,
            max_tokens=150)
        
        total_tokens = response.usage.total_tokens if getattr(response, 'usage', None) else 0

        return {
            "ruling"        : response.choices[0].message.content,
            "usage"         : total_tokens,
            "context_text"  : context_text,
        }
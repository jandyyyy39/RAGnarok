import sys
import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import Config

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = str(BASE_DIR / "chroma_db")

# Singleton
_embeddings = None
_vectorstore = None

def _get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': device}
        )
        _vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=_embeddings
        )
    return _vectorstore

# Exploration-specific SRD topics to bias retrieval toward
EXPLORATION_CONTEXT = (
    "environmental interaction traps locks perception investigation "
    "survival hiding cover stealth searching terrain"
)

def search_exploration(query: str) -> str:
    if not os.path.exists(DB_PATH):
        return "SYSTEM ERROR: ChromaDB not found."

    try:
        vs = _get_vectorstore()

        # Augment the raw query with exploration context so retrieval
        # doesn't compete with combat chunks for ambiguous queries
        augmented_query = f"{query} {EXPLORATION_CONTEXT}"
        results = vs.similarity_search(augmented_query, k=3)

        if not results:
            return "No specific exploration rules found for this action."

        return "\n".join([f"- {doc.page_content}" for doc in results])

    except Exception as e:
        return f"SYSTEM ERROR: Failed to query vector database. {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(search_exploration(" ".join(sys.argv[1:])))
    else:
        print("Error: No query provided.")
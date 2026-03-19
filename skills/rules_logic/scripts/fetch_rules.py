import sys
import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

sys.path.append(str(BASE_DIR))
from config import Config
DB_PATH = str(BASE_DIR / "data/chroma_db")

load_dotenv()

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

def search_rules(query: str) -> str:
    if not os.path.exists(DB_PATH):
        return "SYSTEM ERROR: ChromaDB not found."
    try:
        vs = _get_vectorstore()
        relevant_rules = vs.similarity_search(query, k=3)
        if not relevant_rules:
            return "No specific SRD rules found for this action."
        return "\n".join([f"- {doc.page_content}" for doc in relevant_rules])
    except Exception as e:
        return f"SYSTEM ERROR: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(search_rules(" ".join(sys.argv[1:])))
    else:
        print("Error: No query provided.")
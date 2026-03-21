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
DB_PATH = str(BASE_DIR / "chroma_db")

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

# Exploration-specific SRD topics to bias retrieval toward
EXPLORATION_CONTEXT = (
    # Core Interaction, Traps & Stealth
    "environmental interaction traps locks perception investigation "
    "survival hiding cover half cover three-quarters cover stealth searching secret doors "
    "triggers disarming mechanisms thieves tools "
    
    # Movement, Travel & Navigation
    "terrain difficult terrain travel pace forced march climbing "
    "swimming crawling jumping long jump high jump falling mounts vehicles "
    "navigating getting lost tracking foraging marching order "
    
    # Vision, Light & Senses
    "vision light darkness obscurement illumination darkvision blindsight truesight "
    "dim light bright light heavily obscured lightly obscured invisible unseen "
    
    # Hazards, Environment & Survival Limits
    "hazards weather suffocation choking exhaustion starvation dehydration food water "
    "extreme heat extreme cold strong wind heavy precipitation high altitude "
    
    # Physical Limits & Resting
    "encumbrance lifting carrying capacity pushing dragging size weight "
    "resting short rest long rest downtime hit dice sleep watch "
    
    # Object Manipulation & Destruction
    "objects breaking smashing bursting object armor class AC object hit points HP "
    "doors hinges chains ropes bursting DC"
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
import os
import sys
from pathlib import Path
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Point to the absolute root of RAGnarok to find config and data
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))
from config import Config

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

def search_exploration_rules(query: str, k=2) -> str:
    """
    Retrieves environment, stealth, and survival mechanics from the SRD.
    """
    if not os.path.exists(CHROMA_DIR):
        return "SYSTEM ERROR: ChromaDB not found. DM must wing the exploration rules."
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )
    
    vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    
    # Force the query to look for specific exploration terms to guide the RAG
    augmented_query = f"Rules for: stealth, hiding, perception, investigation, survival, or environment related to: {query}"
    results = vectorstore.similarity_search(augmented_query, k=k)
    
    if not results:
        return "No specific exploration rules found in the SRD for this action."
    
    return "\n\n".join([doc.page_content for doc in results])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        player_intent = " ".join(sys.argv[1:])
        print(search_exploration_rules(player_intent))
    else:
        print("Error: No exploration query provided.")

import os
import sys
from pathlib import Path
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Point to the absolute root of RAGnarok
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))
from config import Config

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

def search_rules(query: str, k=2) -> str:
    if not os.path.exists(CHROMA_DIR):
        return "SYSTEM ERROR: ChromaDB not found. The DM must wing it."
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )
    
    vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    results = vectorstore.similarity_search(query, k=k)
    
    if not results:
        return "No specific rules found in the SRD for this action."
    
    return "\n\n".join([doc.page_content for doc in results])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        player_intent = " ".join(sys.argv[1:])
        print(search_rules(player_intent))
    else:
        print("Error: No query provided.")
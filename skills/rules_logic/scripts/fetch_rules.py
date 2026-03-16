import sys
import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Adjust the path to where Config is located if needed, 
# or just hardcode the embedding model string if Config isn't in scope here.
# Assuming this script is run as a module where config.py is in the root:
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import Config

load_dotenv()

# Dynamically point to the absolute root of RAGnarok where chroma_db lives
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = str(BASE_DIR / "chroma_db")

def search_rules(query: str) -> str:
    """
    Pure data retrieval. NO LLM INFERENCE ALLOWED HERE.
    """
    if not os.path.exists(DB_PATH):
        return "SYSTEM ERROR: ChromaDB not found at expected path. Did you run the ingestion script?"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': device}
        )
        
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
        # Retrieve the most relevant rules from the RAG database
        relevant_rules = vectorstore.similarity_search(query, k=3)
        
        if not relevant_rules:
            return "No specific SRD rules found for this action."
            
        context_text = "\n".join([f"- {doc.page_content}" for doc in relevant_rules])
        return context_text
        
    except Exception as e:
        return f"SYSTEM ERROR: Failed to query vector database. {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(search_rules(" ".join(sys.argv[1:])))
    else:
        print("Error: No query provided.")
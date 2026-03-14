import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

class Config:
    # --- API Keys ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

<<<<<<< Updated upstream
    # --- Models & Parameters ---
    LLM_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DM_TEMPERATURE = 0.7
=======
    # --- Models ---
    # Use LLM_MODELS["default"] for Groq, LLM_MODELS["local"] for Ollama fine-tuned
    LLM_MODELS = {
        "default": "llama-3.3-70b-versatile",
        "fast":    "llama-3.1-8b-instant",
        "local":   "ragnarok-dm",           # Fine-tuned GGUF loaded in Ollama
    }
    LLM_MODEL = LLM_MODELS["default"]      # Shorthand for default Groq model

    # --- Inference ---
    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    # --- Embedding & Generation Parameters ---
    EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
    DM_TEMPERATURE      = 0.7
>>>>>>> Stashed changes
    ARBITER_TEMPERATURE = 0.1

    # --- Cross-Platform Paths ---
    BASE_DIR   = Path(__file__).resolve().parent
    DATA_DIR   = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma_db"

    NPC_DB_PATH = DATA_DIR / "npc_profiles.json"
    STATE_FILE  = DATA_DIR / "world_state.json"
    SRD_FILE    = DATA_DIR / "srd_rules.md"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
<<<<<<< Updated upstream
    
=======
>>>>>>> Stashed changes

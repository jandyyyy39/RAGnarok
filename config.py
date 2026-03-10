import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Silence warnings 
warnings.filterwarnings("ignore")

class Config:
    # --- API Keys ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # --- Models & Parameters ---
    GROQ = {
        'LLM_MODEL': "llama-3.3-70b-versatile",
        'EMBEDDING_MODEL': "all-MiniLM-L6-v2"
    }

    LOCAL = {
        'LLM_MODEL': "qwen3.5:9b",
        'EMBEDDING_MODEL': "Snowflake/snowflake-arctic-embed-m-v1.5"
    }

    DM_TEMPERATURE = 0.7
    ARBITER_TEMPERATURE = 0.1

    # --- Cross-Platform Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma_db"

    # Specific File Paths
    NPC_DB_PATH = DATA_DIR / "npc_profiles.json"
    STATE_FILE  = DATA_DIR / "world_state.json"
    SRD_FILE    = DATA_DIR / "srd_rules.md"

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

class Config:
    # --- API Keys ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN     = os.getenv("HF_TOKEN", None)  # optional — only needed if downloading FIREBALL from HuggingFace

    # --- Models ---
    LLM_MODEL = {
        'GROQ': "llama-3.3-70b-versatile",
        'GROQ_FAST': "llama-3.1-8b-instant",
        'LOCAL': "llama3.2",
        "LOCAL_FAST": "llama3.2",
    }

    # --- Inference ---
    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    # --- Embedding & Generation Parameters ---
    EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
    DM_TEMPERATURE      = 0.7
    ARBITER_TEMPERATURE = 0.1

    # --- Cross-Platform Paths ---
    BASE_DIR   = Path(__file__).resolve().parent
    DATA_DIR   = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma_db"

    NPC_DB_PATH  = DATA_DIR / "npc_profiles.json"
    STATE_FILE   = DATA_DIR / "world_state.json"
    SRD_FILE     = DATA_DIR / "srd_rules.md"
    ESRD_FILE    = DATA_DIR / "5esrd.md"

    # FIREBALL fine-tuning data
    FIREBALL_DIR   = DATA_DIR / "fireball"
    TRAIN_FILE     = DATA_DIR / "fireball_train.jsonl"
    EVAL_FILE      = DATA_DIR / "fireball_eval.jsonl"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

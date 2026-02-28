import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()

class Config:
    # Groq API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Models
    LLM_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # LLM Parameters
    DM_TEMPERATURE = 0.7
    ARBITER_TEMPERATURE = 0.1

    # Base Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

    # File Paths
    NPC_DB_PATH = os.path.join(DATA_DIR, "npc_profiles.json")
    STATE_FILE = os.path.join(DATA_DIR, "world_state.json")
    SRD_FILE = os.path.join(DATA_DIR, "srd_rules.md")

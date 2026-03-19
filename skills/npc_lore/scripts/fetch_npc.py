import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
NPC_FILE = BASE_DIR / "data" / "npc_profiles.json"

def fetch_profile(query: str) -> str:
    if not NPC_FILE.exists():
        return "SYSTEM ERROR: NPC Database missing."
    
    try:
        with open(NPC_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "SYSTEM ERROR: Corrupted NPC JSON."
    
    query_lower = query.lower()
    found_profiles = []
    
    for npc in data.get("npcs", []):
        name = npc.get("name", "").lower()
        first_name = name.split()[0]
        
        if name in query_lower or first_name in query_lower:
            profile_text = (
                f"NPC PROFILE: {npc.get('name', 'Unknown')} ({npc.get('race_and_class', 'Unknown')}). "
                f"Motivation: {npc.get('core_motivation', 'Survival.')} "
                f"Knowledge: {npc.get('knowledge_state', 'Unaware.')} "
                f"Quirks: {npc.get('dialogue_quirks', 'Speaks normally.')} "
                f"Secret: {npc.get('hidden_secret', 'None.')}"
            )
            found_profiles.append(profile_text)
            
    if found_profiles:
        return "\n\n".join(found_profiles)
        
    return "No deep lore retrieved."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(fetch_profile(" ".join(sys.argv[1:])))
    else:
        print("Error: No query provided.")
import json
import sys
import os
from pathlib import Path

# Dynamically point to the absolute root of RAGnarok
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
NPC_FILE = BASE_DIR / "data" / "npc_profiles.json"

def fetch_profile(query: str) -> str:
    """
    Scans the player intent for NPC names and retrieves their JSON profiles.
    """
    if not NPC_FILE.exists():
        return "SYSTEM ERROR: NPC Database missing. Rely on general world state."
    
    try:
        with open(NPC_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "SYSTEM ERROR: Corrupted NPC JSON file. Rely on general world state."
    
    query_lower = query.lower()
    found_profiles = []
    
    # Check every NPC to see if their full name or first name is in the query
    for npc in data.get("npcs", []):
        name = npc.get("name", "").lower()
        first_name = name.split()[0]
        
        if name in query_lower or first_name in query_lower:
            # Flatten the JSON into dense semantic text for token efficiency
            profile_text = (
                f"NPC PROFILE: {npc.get('name')} ({npc.get('race')} {npc.get('class')}). "
                f"Personality: {npc.get('personality')} "
                f"Quirks: {npc.get('dialogue_quirks')} "
                f"Secret: {npc.get('secret')}"
            )
            found_profiles.append(profile_text)
            
    if found_profiles:
        return "\n\n".join(found_profiles)
        
    return "No specific NPC profile requested. Rely on the general world state."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        intent = " ".join(sys.argv[1:])
        print(fetch_profile(intent))
    else:
        print("Error: No query provided.")
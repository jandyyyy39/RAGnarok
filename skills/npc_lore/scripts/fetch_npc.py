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

    # Intent-aware field selection
    query_lower = query.lower()

    SOCIAL_TRIGGERS    = ["talk", "ask", "say", "tell", "whisper", "persuade", "convince", "greet"]
    COMBAT_TRIGGERS    = ["attack", "lunge", "strike", "grab", "shove", "fight", "threaten", "intimidate"]
    SECRET_TRIGGERS    = ["persuasion", "intimidation", "insight", "bribe", "interrogate"]

    is_social  = any(t in query_lower for t in SOCIAL_TRIGGERS)
    is_combat  = any(t in query_lower for t in COMBAT_TRIGGERS)
    is_secret  = any(t in query_lower for t in SECRET_TRIGGERS)

    # Always include identity. Select extras by intent.
    def build_profile(npc: dict) -> str:
        parts = [f"{npc.get('name')} ({npc.get('race_and_class')})"]

        if is_social or is_secret:
            parts.append(f"Motivation: {npc.get('core_motivation')}")

        if is_combat:
            parts.append(f"Motivation: {npc.get('core_motivation')}")  # affects fight-or-flee


        if is_secret:
            parts.append(f"Secret (do not reveal directly): {npc.get('hidden_secret')}")

        # Knowledge_state only if the NPC might reference the player
        if is_social or is_combat:
            parts.append(f"Knowledge: {npc.get('knowledge_state')}")

        # Always add dialogue quirks at the end if any, since it can flavor all interactions
        parts.append(f"Quirks: {npc.get('dialogue_quirks')}")

        return " | ".join(parts)

    found_profiles = []
    for npc in data.get("npcs", []):
        name = npc.get("name", "").lower()
        first_name = name.split()[0]
        if name in query_lower or first_name in query_lower:
            found_profiles.append(build_profile(npc))

    if found_profiles:
        return "\n\n".join(found_profiles)

    return "No deep lore retrieved."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(fetch_profile(" ".join(sys.argv[1:])))
    else:
        print("Error: No query provided.")
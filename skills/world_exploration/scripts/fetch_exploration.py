import sys

def search_exploration(query: str) -> str:
    query_lower = query.lower()
    
    if "search" in query_lower or "look for" in query_lower or "hidden" in query_lower:
        return "RULE: Finding hidden objects or clues requires an Intelligence (Investigation) or Wisdom (Perception) check."
    if "lock" in query_lower or "pick" in query_lower or "disarm" in query_lower:
        return "RULE: Picking a lock or disarming a trap requires a Dexterity (Sleight of Hand) check, usually with Thieves' Tools."
    if "track" in query_lower or "forage" in query_lower or "navigate" in query_lower:
        return "RULE: Tracking, foraging, or navigating wilderness requires a Wisdom (Survival) check."
        
    return "RULE: General environmental interaction. Roll 1d20 against a DC determined by the DM."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(search_exploration(" ".join(sys.argv[1:])))
    else:
        print("Error: No query provided.")
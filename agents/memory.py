import json
import os

class MemoryAgent:
    def __init__(self, state_file="data/world_state.json"):
        self.state_file = state_file
        self.default_state = {
            "current_location": "Unknown",
            "active_npcs": [],
            "party_status": "Healthy",
            "recent_events": [],
            "inventory_changes": []
        }
        self.initialize_state()

    def initialize_state(self):
        """Creates the state file if it doesn't exist."""
        if not os.path.exists(self.state_file):
            with open(self.state_file, "w") as f:
                json.dump(self.default_state, f, indent=4)

    def get_current_state(self):
        with open(self.state_file, "r") as f:
            return json.load(f)

    def update_state(self, new_data: dict):
        """
        Updates specific fields in the world state.
        Example new_data: {"current_location": "The Goblin Cave"}
        """
        state = self.get_current_state()
        state.update(new_data)
        
        # Keep only the last 5 events to prevent context bloat
        if "recent_events" in new_data:
            state["recent_events"] = state["recent_events"][-5:]
            
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=4)

    def format_for_dm(self):
        """Converts the JSON state into a narrative string for the DM Agent."""
        state = self.get_current_state()
        recap = f"""
        CURRENT WORLD STATE:
        - Location: {state['current_location']}
        - Active NPCs: {', '.join(state['active_npcs']) if state['active_npcs'] else 'None'}
        - Party Condition: {state['party_status']}
        - Recent History: {' '.join(state['recent_events'])}
        """
        return recap

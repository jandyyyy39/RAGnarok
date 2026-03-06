from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from agents.safety import SafetyAgent
from agents.memory import MemoryAgent
from agents.rules_arbiter import RulesArbiter
from agents.dungeon_master import DMAgent
from agents.npc_consistency import NPCConsistencyAgent

class RAGnarokOrchestrator:
    def __init__(self):
        print("Initializing RAGnarok Multi-Agent System...")
        self.safety = SafetyAgent()
        self.memory = MemoryAgent()
        self.arbiter = RulesArbiter()
        self.dm = DMAgent()
        self.npc_agent = NPCConsistencyAgent()
        print("All agents online. Ready to play.\n")

    def process_turn(self, player_input: str):
        print("\n" + "="*50)
        
        # 1. Safety Check
        if not self.safety.check(player_input):
            return "Safety Agent Intercept: That action violates the table's safety tools."

        # 2. Retrieve World State
        print("[Memory Agent] Fetching recap...")
        world_state = self.memory.format_for_dm()

        # 3. Rules Arbiter Assessment
        print("[Rules Arbiter] Consulting the SRD...")
        ruling = self.arbiter.get_ruling(player_input, world_state)
        print(f"  -> Arbiter says: {ruling[:100]}...")

        # 4. Narrative Generation
        print("[DM Agent] Weaving the narrative...")
        dm_response = self.dm.generate_response(player_input, world_state, ruling)

        # 5. NPC Consistency Pass
        print("[NPC Agent] Checking character sheets...")
        final_output = self.npc_agent.refine_dialogue(dm_response)

        # 6. Update Memory
        current_events = self.memory.get_current_state()["recent_events"]
        new_event = f"Player: {player_input} | Outcome: {final_output[:100]}..." # Truncated to save token space
        
        self.memory.update_state({
            "recent_events": current_events + [new_event]
        })

        print("="*50 + "\n")
        return final_output

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
game = RAGnarokOrchestrator()

@app.route('/api/game', methods=['POST'])
def handle_game_turn():
    data = request.json
    player_input = data.get('input')
    if not player_input:
        return jsonify({'error': 'No input provided'}), 400

    response = game.process_turn(player_input)
    game_state = game.memory.get_current_state()
    return jsonify({'response': response, 'game_state': game_state})

if __name__ == "__main__":
    game.memory.update_state({
        "current_location": "The Yawning Portal Tavern",
        "active_npcs": ["Durnan the Barkeep"],
        "recent_events": ["The party just walked into the crowded tavern."]
    })
    app.run(port=5000, debug=True)
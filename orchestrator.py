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
        print ("\n" + "="*50)

        # --- INTERCEPTOR ---
        is_system_roll = player_input.startswith("[SYSTEM: ROLL_RESOLUTION")

        if is_system_roll:
            print("[Orchestrator] Dice Roll detected. Bypassing Arbiter.")
            world_state = self.memory.format_for_dm()

            ruling = f"SYSTEM OVERRIDE: The player has rolled the dice. Data: {player_input}. You must narrate the exact outcome of this result based on whether it is a SUCCESS or FAILURE. Do not ask for another roll."
        else: 
            # --- NORMAL TURN ---
            # Safety Check
            if not self.safety.check(player_input):
                return {"response": "Safety Agent Intercept: That action violates the table's safety tools.", "pending_action": None}
            
            # Retrieve world state
            print("[Memory Agent] Fetching recap...")
            world_state = self.memory.format_for_dm()

            # Rules arbiter assessment
            print("[Rules Arbiter] Consulting the SRD...")
            ruling = self.arbiter.get_ruling(player_input, world_state)
            print(f"[Rules Arbiter] Ruling: {ruling}")
        
        # DM generates response
        print("[DM Agent] Weaving the narrative...")
        dm_result = self.dm.generate_response(player_input, world_state, ruling)

        # --- FORK ---
        if dm_result["type"] == "tool_call":
            print(f"[DM Agent] Halting narrative. Requesting {dm_result['action']['skill']} check.")
            return {
                "response": dm_result["narrative"],
                "pending_action": dm_result["action"]
            }

        print("[NPC Agent] Checking character sheets...")
        final_output = self.npc_agent.refine_dialogue(dm_result["narrative"])
        
        # Memory update
        current_events = self.memory.get_current_state()["recent_events"]
        
        event_text = player_input if not is_system_roll else f"The player rolled dice. {player_input}"
        new_event = f"Action: {event_text} | Outcome: {final_output[:100]}..." 
        
        self.memory.update_state({"recent_events": current_events + [new_event]})

        print("="*50 + "\n")
        return {
            "response": final_output,
            "pending_action": None
        }
    
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
game = RAGnarokOrchestrator()

@app.route('/api/game', methods=['POST'])
def handle_game_turn():
    data = request.json
    player_input = data.get('input')
    if not player_input:
        return jsonify({'error': 'No input provided'}), 400

    turn_result = game.process_turn(player_input)
    
    return jsonify({
        'response': turn_result["response"],
        'pending_action': turn_result["pending_action"],
        'game_state': game.memory.get_current_state()
    })

if __name__ == "__main__":
    game.memory.update_state({
        "current_location": "The Yawning Portal Tavern",
        "active_npcs": ["Durnan the Barkeep"],
        "recent_events": ["The party just walked into the crowded tavern."]
    })
    app.run(port=5000, debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from agents.safety import SafetyAgent
from agents.memory import MemoryAgent
from agents.rules_arbiter import RulesArbiter
from agents.dungeon_master import DMAgent
from agents.npc_consistency import NPCConsistencyAgent
import os
import json
from datetime import datetime
import torch
import argparse
from groq import Groq
from openai import OpenAI
from time import time

class RAGnarokOrchestrator:
    def __init__(self, client, model_profile: str):
        print(f"Initializing RAGnarok Multi-Agent System with [{model_profile}] settings...")
        self.safety = SafetyAgent()
        self.memory = MemoryAgent()
        self.arbiter = RulesArbiter(client=client, model_profile=model_profile)
        self.dm = DMAgent(client=client, model_profile=model_profile)
        self.npc_agent = NPCConsistencyAgent(client=client, model_profile=model_profile)
        
        # --- SESSION GENERATOR ---
        os.makedirs("data/history", exist_ok=True) 
        
        # session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"data/history/baseline_architecture_log.json" 
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump([], f)
            
        print(f"Session telemetry initialized: {self.log_file}")
        print("All agents online. Ready to play.\n")

    def process_turn(self, player_input: str, args):
        start_time = time()
        turn_log = []
        def trace(message: str):
            print(message)
            turn_log.append(message)

        trace("\n" + "="*50)

        # --- INTERCEPTOR ---
        is_system_roll = player_input.startswith("[SYSTEM: ROLL_RESOLUTION")
        
        # Standardize Baseline Routes: In this architecture, everything is ALWAYS on.
        baseline_routes = {
            "rules_logic": True, 
            "npc_lore": True, 
            "world_exploration": True
        }

        if is_system_roll:
            trace("[Orchestrator] 🎲 Dice Roll detected. Bypassing Arbiter.")
            world_state = self.memory.format_for_dm()
            recent_history = self.memory.get_current_state()["recent_events"]
            last_action = "unknown action"
            for event in reversed(recent_history):
                if event.startswith("Player:"):
                    last_action = event.split("Player: ")[1].split(" | ")[0]
                    break

            ruling = f"SYSTEM OVERRIDE: The player was trying to: '{last_action}'. Roll: {player_input}."
        else: 
            # --- NORMAL TURN ---
            if not self.safety.check(player_input):
                return {"response": "Safety Agent Intercept.", "pending_action": None}
            
            world_state = self.memory.format_for_dm()
            trace("[Rules Arbiter] Consulting the SRD...")
            ruling = self.arbiter.get_ruling(player_input, world_state)
        
        # DM generates response
        trace("[DM Agent] Weaving the narrative...")
        dm_result = self.dm.generate_response(player_input, world_state, ruling)

        # Handle Output
        if dm_result["type"] == "tool_call":
            final_output = dm_result["narrative"]
            pending_action = dm_result["action"]
        else:
            trace("[NPC Agent] Checking character sheets...")
            final_output = self.npc_agent.refine_dialogue(dm_result["narrative"])
            pending_action = None
        
        # Memory update
        new_event = f"Action: {player_input} | Outcome: {final_output[:100]}..." 
        self.memory.update_state({"recent_events": self.memory.get_current_state()["recent_events"] + [new_event]})

        # --- TELEMETRY LOGGING (Thesis Metrics) ---
        latency_ms = (time() - start_time) * 1000
        
        self._save_history({
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "architecture": "baseline",
            "latency_ms": round(latency_ms),
            "total_tokens": 0,  # Placeholder
            "routing_hallucination": False,
            "routes_fired": baseline_routes,
            "retrieval_types": {
                "rag_vector": True,
                "kv_lookup": True
            },
            "async_critic_scores": {
                "rule_accuracy": 0, 
                "narrative_coherence": 0, 
                "npc_voice_consistency": 0
            },
            "player_intent": player_input,
            "final_output": {"response": final_output, "pending_action": pending_action}
        })

        return {"response": final_output, "pending_action": pending_action}

    def _save_history(self, turn_data: dict):
        """Standardized JSON log for comparison."""
        with open(self.log_file, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except:
                history = []
        history.append(turn_data)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow all origins for development
game = None # Will be initialized in main

@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    if not game:
        return jsonify({"error": "Game not initialized"}), 500
    return jsonify(game.memory.get_current_state())

@app.route('/api/game', methods=['POST'])
def handle_game_turn():
    if not game:
        return jsonify({"error": "Game not initialized"}), 500
    data = request.json
    player_input = data.get('input')
    if not player_input:
        return jsonify({'error': 'No input provided'}), 400

    args = app.config['args']
    turn_result = game.process_turn(player_input, args)
    
    return jsonify({
        'response': turn_result["response"],
        'pending_action': turn_result["pending_action"],
        'game_state': game.memory.get_current_state()
    })


if __name__ == "__main__":
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SYSTEM: Activating on **{device.upper()}**.")

    parser = argparse.ArgumentParser(description="RAGnarok Orchestrator")
    parser.add_argument(
        "-p", "--profile",
        choices=["local", "fast", "default"],
        default="default",
        help="Set the profile"
    )
    parser.add_argument('--no-rag', action='store_true', help="Skip Rules Arbiter entirely.")
    parser.add_argument('--no-memory', action='store_true', help="Don't inject world state.")
    parser.add_argument('--no-npc-const', action='store_true', help="Skip NPC Consistency step.")
    args = parser.parse_args()
    app.config['args'] = args
    profile = args.profile

    if profile == "default":
        print("Using Groq API...")
        client = Groq(api_key=Config.GROQ_API_KEY)
        model_profile = "GROQ"
    elif profile == "local":
        print("Using local model...")
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model_profile = "LOCAL"
    elif profile == "fast":
        print("Using fast model...")
        client = Groq(api_key=Config.GROQ_API_KEY)
        model_profile = "GROQ_FAST"

    game = RAGnarokOrchestrator(client=client, model_profile=model_profile)
    
    # Initialize the world with NPCs
    game.memory.update_state({
        "current_location": "The Black Boar Tavern",
        "active_npcs": [
            "Thrain Blackbeard (Human Barbarian)", 
            "Elara Moonwhisper (Half-Elf Bard)",
            "Arin the Bold (Human Rogue)"
        ],
        "recent_events": ["The party just walked into the loud, sea-shanty-filled Black Boar Tavern. Thrain is yelling for stronger ale, while Elara sings a haunting melody. What would you like to do?"]
    })
    app.run(port=5000, debug=True, use_reloader=False)
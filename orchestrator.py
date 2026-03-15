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
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"data/history/session_{session_id}.json" 
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump([], f)
            
        print(f"Session telemetry initialized: {self.log_file}")
        print("All agents online. Ready to play.\n")

    def process_turn(self, player_input: str, args):
        turn_log = []
        def trace(message: str):
            print(message)
            turn_log.append(message)

        trace("\n" + "="*50)

        # --- INTERCEPTOR ---
        is_system_roll = player_input.startswith("[SYSTEM: ROLL_RESOLUTION")

        if is_system_roll:
            trace("[Orchestrator] 🎲 Dice Roll detected. Bypassing Arbiter.")
            world_state = self.memory.format_for_dm()
            
            # --- Look at the last REAL player action in memory ---
            recent_history = self.memory.get_current_state()["recent_events"]
            # Get the last event that starts with "Player:"
            last_action = "unknown action"
            for event in reversed(recent_history):
                if event.startswith("Player:"):
                    last_action = event.split("Player: ")[1].split(" | ")[0]
                    break

            # Force the DM to tie the roll result to that specific action
            ruling = f"""
            SYSTEM OVERRIDE: 
            The player was trying to: "{last_action}"
            The roll result is: {player_input}
            
            NARRATION RULE: You MUST narrate the outcome of "{last_action}". 
            - If SUCCESS: Describe how the player achieves their goal, what they learn, or how the environment reacts favorably.
            - If FAILURE: Describe the negative consequences, the NPC's refusal, or the mechanical setback.
            Do NOT mention the underlying math in the narrative.
            """
        else: 
            # --- NORMAL TURN ---
            # Safety Check
            if not self.safety.check(player_input):
                return {"response": "Safety Agent Intercept: That action violates the table's safety tools.", "pending_action": None}
            
            # Retrieve world state
            if args.no_memory:
                trace("[Orchestrator] --no-memory: Skipping world state injection.")
                world_state = "No world state provided by the Memory agent."
            else:
                trace("[Memory Agent] Fetching recap...")
                world_state = self.memory.format_for_dm()

            # Rules arbiter assessment
            if args.no_rag:
                trace("[Orchestrator] --no-rag: Skipping RAG lookup.")
                ruling = "The Rules Arbiter was not consulted."
            else:
                trace("[Rules Arbiter] Consulting the SRD...")
                ruling = self.arbiter.get_ruling(player_input, world_state)
                trace(f"[Rules Arbiter] Ruling: {ruling}")
        
        # DM generates response
        trace("[DM Agent] Weaving the narrative...")
        dm_result = self.dm.generate_response(player_input, world_state, ruling)

        # --- FORK ---
        if dm_result["type"] == "tool_call":
            trace(f"[DM Agent] Halting narrative. Requesting {dm_result['action']['skill']} check.")
            result = {
                "response": dm_result["narrative"],
                "pending_action": dm_result["action"]
            }
            self._save_history({
                "timestamp": datetime.now().isoformat(),
                "user_input": player_input,
                "backend_log": "\n".join(turn_log),
                "final_output": result
            })
            return result

        if args.no_npc_const:
            trace("[Orchestrator] --no-npc-const: Skipping NPC consistency check.")
            final_output = dm_result["narrative"]
        else:
            trace("[NPC Agent] Checking character sheets...")
            final_output = self.npc_agent.refine_dialogue(dm_result["narrative"])
        
        # Memory update
        current_events = self.memory.get_current_state()["recent_events"]
        
        event_text = player_input if not is_system_roll else f"The player rolled dice. {player_input}"
        new_event = f"Action: {event_text} | Outcome: {final_output[:100]}..." 
        
        self.memory.update_state({"recent_events": current_events + [new_event]})

        trace("="*50 + "\n")
        result = {
            "response": final_output,
            "pending_action": None
        }

        self._save_history({
            "timestamp": datetime.now().isoformat(),
            "user_input": player_input,
            "backend_log": "\n".join(turn_log),
            "final_output": result
        })

        return result
    
    def _save_history(self, turn_data: dict):
        """Appends a structured turn record to the JSON flight recorder."""
        with open(self.log_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        # Append new turn
        history.append(turn_data)
        
        # Save it back
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
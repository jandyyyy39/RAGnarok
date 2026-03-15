import json
import os
import time
import re
from datetime import datetime
from typing import Dict, Any, List
import argparse 
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from openai import OpenAI
from config import Config
from skills.combat_mechanics.scripts import execute_rag_search
from skills.social_lore.scripts import fetch_npc_profile
from skills.exploration_survival.scripts import execute_exploration
from agents.dm_graph import DMGraphAgent
from agents.memory import MemoryAgent
from agents.supervisor import SupervisorAgent

# --- SKILL METADATA FUNCTIONS ---
def get_skill_metadata() -> Dict[str, str]:
    skills_dir = "skills"
    metadata = {}
    for skill_name in os.listdir(skills_dir):
        skill_path = os.path.join(skills_dir, skill_name)
        if os.path.isdir(skill_path):
            skill_file = os.path.join(skill_path, "SKILL.md")
            if os.path.exists(skill_file):
                with open(skill_file, 'r') as f:
                    content = f.read()
                    metadata[skill_name] = content.split('---')[0].strip()
    return metadata

def get_skill_instructions(skill_name: str) -> str:
    skill_file = os.path.join("skills", skill_name, "SKILL.md")
    with open(skill_file, 'r') as f:
        content = f.read()
        return content.split('---')[1].strip()

# --- CRITIC AGENT ---
def critic_evaluator_agent(narrative: str) -> Dict[str, Any]:
    print("--- CRITIC: Evaluating draft ---")
    meta_game_terms = re.compile(r'\b(DC|skill check|roll|saving throw|attack roll)\b', re.IGNORECASE)
    found_terms = meta_game_terms.findall(narrative)
    if found_terms:
        feedback = f"Found meta-game terms: {', '.join(list(set(found_terms)))}. Rewrite to be purely narrative."
        print(f"Critic: REJECTED. {feedback}")
        return {"is_approved": False, "feedback": feedback}
    print("Critic: APPROVED.")
    return {"is_approved": True, "feedback": "Approved."}

# --- THE GRAPH ORCHESTRATOR CLASS ---
class RAGnarokGraphOrchestrator:
    def __init__(self, client, model_profile: str):
        print(f"Initializing EXPERIMENTAL Graph System [{model_profile}]...")
        self.memory = MemoryAgent()
        self.supervisor = SupervisorAgent(client=client, model_profile=model_profile)
        self.dm = DMGraphAgent(client=client, model_profile=model_profile)
        self.skills_metadata = get_skill_metadata()
        
        os.makedirs("data/history", exist_ok=True) 
        self.log_file = "data/history/graph_architecture_log.json"
        
        # Ensure log file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def process_turn(self, player_input: str, args):
        start_time = time.time()
        print("\n" + "="*50)
        
        # 1. State Blackboard Initialization
        shared_state: Dict[str, Any] = {
            "player_intent": player_input,
            "active_skill": None,
            "retrieved_context": "",
            "draft_narrative": "",
            "error_feedback": "",
            "final_response": None,
            "critic_rejections": 0
        }

        # --- DICE ROLL INTERCEPTOR (Bypasses Graph) ---
        is_system_roll = player_input.startswith("[SYSTEM: ROLL_RESOLUTION")
        world_state = self.memory.format_for_dm()

        if is_system_roll:
            print("[Orchestrator] Dice Roll detected. Bypassing Supervisor.")
            
            # Reconstruct context for DM
            recent_history = self.memory.get_current_state()["recent_events"]
            last_action = "unknown action"
            for event in reversed(recent_history):
                if event.startswith("Action:"):
                    last_action = event.split("Action: ")[1].split(" | ")[0]
                    break

            current_active_npcs = self.memory.get_current_state().get("active_npcs", [])
            
            # NPC
            npc_query_string = " ".join(current_active_npcs)
            if npc_query_string.strip():
                npc_context = fetch_npc_profile.fetch_profile(npc_query_string)
            else:
                npc_context = "No known NPCs are currently in the immediate vicinity."

            ruling = f"SYSTEM OVERRIDE: The player was trying to: '{last_action}'. The roll result is: {player_input}. Narrate the outcome."
            
            # resolution_prompt = (
            #     "You are an immersive, theatrical Dungeon Master narrating the outcome of a dice roll. "
            #     "Embody the environment and the NPCs. Make the scene dramatic and reactive. "
            #     "Push the narrative forward. Do NOT just describe people staring. "
            #     "Do NOT mention math or dice mechanics."
            # )
            resolution_prompt = (
                "You are an immersive, theatrical Dungeon Master narrating the outcome of a dice roll. "
                "Embody the environment and the NPCs. "
                "Based on the success or failure of the roll, you MUST drive the narrative forward. "
                "A successful roll means the player achieves their goal: an NPC agrees to a deal, reveals a secret, or a physical obstacle is overcome. "
                "A failed roll means a setback: an NPC rejects them, a trap triggers, or they are attacked. "
                "Do NOT stall. Do NOT just describe the room reacting passively. Force a concrete consequence. "
                "Do NOT mention math or dice mechanics."
            )

            dm_result = self.dm.generate_response(
                system_prompt=resolution_prompt,
                player_input=player_input,
                world_state=world_state,
                context=ruling + f"\n\nNPC PROFILES:\n{npc_context}"
            )
            shared_state["final_response"] = dm_result
            shared_state["active_skill"] = "system_roll"

        else:
            # --- NORMAL GRAPH ROUTING ---
            # 2. Supervisor Routing
            active_skill = self.supervisor.route_intent(player_input, self.skills_metadata)
            shared_state["active_skill"] = active_skill

            # 3. Execute Skill (Level 3)
            if active_skill == "combat_mechanics":
                shared_state["retrieved_context"] = execute_rag_search.search_rules(player_input)
            elif active_skill == "social_lore":
                shared_state["retrieved_context"] = fetch_npc_profile.fetch_profile(player_input)
            elif active_skill == "exploration_survival":
                shared_state["retrieved_context"] = execute_exploration.search_exploration_rules(player_input)
            else:
                shared_state["retrieved_context"] = "No specific rules required."

            # 4. DM + Critic Loop
            dm_prompt = get_skill_instructions(active_skill) 
            critic_feedback = ""
            max_retries = 2
            
            for i in range(max_retries + 1):
                dm_result = self.dm.generate_response(
                    system_prompt=dm_prompt, 
                    player_input=player_input,
                    world_state=world_state,
                    context=shared_state["retrieved_context"],
                    critic_feedback=critic_feedback
                )

                if dm_result["type"] == "tool_call":
                    shared_state["final_response"] = dm_result
                    break

                draft = dm_result["narrative"]
                evaluation = critic_evaluator_agent(draft)

                if evaluation["is_approved"]:
                    shared_state["final_response"] = dm_result
                    break
                else:
                    shared_state["critic_rejections"] += 1
                    critic_feedback = f"Attempt {i+1} Failed. {evaluation['feedback']}"
                    if i == max_retries:
                        shared_state["final_response"] = {"type": "text", "narrative": f"(System Intervention: {draft})"}

        # --- END OF TURN LOGIC ---
        latency_ms = (time.time() - start_time) * 1000
        
        # Check if the response is a tool call or just text
        is_tool = shared_state["final_response"].get("type") == "tool_call"
        
        # If it's a tool, use the placeholder narrative; otherwise use the full text
        final_output_text = shared_state["final_response"].get("narrative", "")
        
        # Memory Update (We only want to store the text in history)
        current_events = self.memory.get_current_state()["recent_events"]
        new_event = f"Action: {player_input} | Outcome: {final_output_text[:100]}..." 
        self.memory.update_state({"recent_events": current_events + [new_event]})

        # Structure the API return
        result = {
            "response": final_output_text,
            "pending_action": shared_state["final_response"].get("action") if is_tool else None
        }
        
        # Memory Update
        current_events = self.memory.get_current_state()["recent_events"]
        event_text = player_input if not is_system_roll else f"The player rolled dice. {player_input}"
        new_event = f"Action: {event_text} | Outcome: {final_output_text[:100]}..." 
        self.memory.update_state({"recent_events": current_events + [new_event]})

        # Structure the API return
        result = {
            "response": final_output_text,
            "pending_action": shared_state["final_response"].get("action") if shared_state["final_response"]["type"] == "tool_call" else None
        }

        # Telemetry Logging
        telemetry = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "architecture": "progressive_disclosure_graph",
            "latency_ms": round(latency_ms),
            "total_tokens": 0,
            "route_taken": shared_state["active_skill"],
            "critic_rejections": shared_state["critic_rejections"],
            "player_intent": player_input,
            "final_output": result
        }
        self._save_history(telemetry)
        
        print(f"--- Turn Complete | Latency: {telemetry['latency_ms']}ms | Route: {telemetry['route_taken']} ---")
        return result

    def _save_history(self, turn_data: dict):
        with open(self.log_file, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
        history.append(turn_data)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) 
game = None 

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
    parser = argparse.ArgumentParser(description="Experimental Graph Orchestrator")
    parser.add_argument('--local', action='store_true', help="Use local Ollama instead of Groq.")
    args = parser.parse_args()
    app.config['args'] = args

    if args.local:
        print("Using local model...")
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model_profile = "LOCAL"
    else:
        print("Using Groq API...")
        client = Groq(api_key=Config.GROQ_API_KEY)
        model_profile = "GROQ"

    game = RAGnarokGraphOrchestrator(client=client, model_profile=model_profile)
    
    # Initialize the world state
    game.memory.update_state({
        "current_location": "The Black Boar Tavern",
        "active_npcs": [
            "Thrain Blackbeard (Human Barbarian)", 
            "Elara Moonwhisper (Half-Elf Bard)",
            "Arin the Bold (Human Rogue)"
        ],
        "recent_events": ["The party just walked into the loud, sea-shanty-filled Black Boar Tavern. Thrain is yelling for stronger ale, while Elara sings a haunting yet beautiful melody in the corner. What would you like to do?"]
    })
    app.run(port=5000, debug=True, use_reloader=False)
    
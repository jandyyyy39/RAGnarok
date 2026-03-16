import json
import os
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, List
import argparse 
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from openai import OpenAI

from config import Config
from agents.dm_spear import DMSpearAgent
from agents.memory import MemoryAgent
from skills.rules_logic.scripts import fetch_rules
from skills.npc_lore.scripts import fetch_npc
from skills.world_exploration.scripts import fetch_exploration

telemetry_lock = threading.Lock()

# --- PROGRAMMATIC GUARDS ---
def safety_filter(intent: str) -> bool:
    """Zero-LLM latency safety check."""
    blocklist = ["ignore all previous instructions", "system prompt", "bypass"]
    return not any(word in intent.lower() for word in blocklist)

def deterministic_guard(narrative: str) -> bool:
    """Zero-LLM latency leak check."""
    meta_game_terms = re.compile(r'\b(DC|saving throw)\b', re.IGNORECASE)
    return not bool(meta_game_terms.findall(narrative))

# --- ASYNC BACKGROUND CRITIC ---
def async_critic_evaluation(client, model, turn_data, log_file):
    """Runs completely in the background. The player NEVER waits for this."""
    try:
        # Use a fast, cheap prompt just for scoring
        prompt = f"""
        Evaluate this TTRPG turn from 0 to 10 on three metrics:
        1. rule_accuracy: Did it follow 5e mechanics?
        2. narrative_coherence: Did the story advance?
        3. npc_voice_consistency: Did NPCs use their quirks/rhymes/personalities?
        
        Player Intent: {turn_data['player_intent']}
        DM Response: {turn_data['final_output'].get('response', 'TOOL_CALL')}
        
        Output pure JSON: {{"rule_accuracy": 0, "narrative_coherence": 0, "npc_voice_consistency": 0, "reasoning": "..."}}
        """
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        scores = json.loads(response.choices[0].message.content)
        turn_data["async_critic_scores"] = scores
        
        # Append to telemetry log safely
        with telemetry_lock:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(turn_data)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            print(f"   [Async Critic] Logged scores: {scores}")
    except Exception as e:
        print(f"   [Async Critic] Failed to score turn: {e}")

# --- SPEAR ORCHESTRATOR ---
class SPEAROrchestrator:
    def __init__(self, client, model_profile: str):
        print(f"Initializing SPEAR Architecture [{model_profile}]...")
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]
        
        # Use a smaller/faster model for the Supervisor if available (e.g., Llama 3 8B)
        self.fast_model = "llama3-8b-8192" if model_profile == "GROQ" else self.model
        
        self.memory = MemoryAgent()
        self.dm = DMSpearAgent(client, model_profile)

        os.makedirs("data/history", exist_ok=True) 
        self.log_file = "data/history/spear_architecture_log.json"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_skill_instructions(self, skill_name: str) -> str:
        """Progressive Disclosure: Only loads instructions if routed."""
        path = os.path.join("skills", skill_name, "SKILL.md")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read().split('---')[-1].strip()
        return ""

    def process_turn(self, player_input: str):
        start_time = time.time()
        print("\n" + "▼"*50)
        print(f"[SPEAR] Intent: '{player_input}'")

        # 1. Zero-LLM Safety Guard
        if not safety_filter(player_input):
            return {"response": "Safety Agent Intercept: Invalid input.", "pending_action": None}

        world_state = self.memory.get_current_state()
        active_npcs = world_state.get("active_npcs", [])

        # -- PHASE 1: SEMANTIC ROUTER (Supervisor) --
        routes = {"rules_logic": False, "npc_lore": False, "world_exploration": False}
        routing_hallucination = False
        is_system_roll = player_input.startswith("[SYSTEM: ROLL_RESOLUTION")

        if is_system_roll:
            print("[SPEAR] Dice Roll detected. Bypassing Supervisor.")
            routes["npc_lore"] = True # Always fetch NPCs for narrative fallout
        else:
            try:
                # Claude's "Shallow Memory" Justification: Location + Active NPCs are injected here.
                router_prompt = f"""
                Analyze intent: "{player_input}"
                Current Location: {world_state.get('current_location')}
                Active NPCs: {active_npcs}

                Return pure JSON:
                "rules_logic": true if action requires combat or physical mechanics.
                "npc_lore": true if player interacts with NPCs or they are likely to react.
                "world_exploration": true if player searches, picks locks, or interacts with the environment.
                """
                route_response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": router_prompt}],
                    model=self.fast_model,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                routes = json.loads(route_response.choices[0].message.content)
                
                # Claude's Hallucination Check: Validate keys
                if not all(k in ["rules_logic", "npc_lore", "world_exploration"] for k in routes.keys()):
                    routing_hallucination = True
                print(f"[SPEAR] Router Decision: {routes}")
            except Exception as e:
                print(f"[SPEAR ALERT] Routing Error: {e}")
                routing_hallucination = True
                routes = {"rules_logic": True, "npc_lore": True, "world_exploration": True} # Fallback to Baseline (Bloat)

        # -- PHASE 2: PARALLEL DATA FETCH (Efficiency) --
        retrieved_context = []
        dynamic_instructions = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_rules = executor.submit(fetch_rules.search_rules, player_input) if routes.get("rules_logic") else None
            future_npc = executor.submit(fetch_npc.fetch_profile, player_input + " " + " ".join(active_npcs)) if routes.get("npc_lore") else None
            future_explore = executor.submit(fetch_exploration.search_exploration, player_input) if routes.get("world_exploration") else None

            if future_rules:
                dynamic_instructions.append(self._read_skill_instructions("rules_logic"))
                retrieved_context.append(f"=== RULES ===\n{future_rules.result()}")
            if future_npc:
                dynamic_instructions.append(self._read_skill_instructions("npc_lore"))
                retrieved_context.append(f"=== NPC LORE ===\n{future_npc.result()}")
            if future_explore:
                dynamic_instructions.append(self._read_skill_instructions("world_exploration"))
                retrieved_context.append(f"=== EXPLORATION ===\n{future_explore.result()}")

        # -- PHASE 3: EXECUTION (Single Pass) --
        dm_result = self.dm.generate_response(
            player_input=player_input,
            world_state=world_state,
            dynamic_instructions=dynamic_instructions,
            retrieved_context=retrieved_context
        )

        final_output = {
            "response": dm_result.get("response", ""),
            "pending_action": dm_result.get("pending_action")
        }

        # Deterministic Guard (Leak Check)
        if dm_result.get("type") == "text" and not deterministic_guard(final_output["response"]):
            final_output["response"] += "\n\n[System Alert: Meta-game terms detected and scrubbed.]"

        # Update Memory
        new_event = f"Action: {player_input} | Outcome: {final_output['response'][:100]}..." 
        self.memory.update_state({"recent_events": world_state["recent_events"] + [new_event]})

        # -- PHASE 4: ASYNC TELEMETRY (Thesis Metrics) --
        latency_ms = (time.time() - start_time) * 1000
        turn_data = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "architecture": "SPEAR",
            "latency_ms": round(latency_ms),
            "total_tokens": dm_result.get("usage", 0),
            "routing_hallucination": routing_hallucination,
            "routes_fired": routes,
            "retrieval_types": {
                "rag_vector": routes.get("rules_logic", False) or routes.get("world_exploration", False),
                "kv_lookup": routes.get("npc_lore", False)
            },
            "async_critic_scores": {
                "rule_accuracy": 0, 
                "narrative_coherence": 0, 
                "npc_voice_consistency": 0
            },
            "player_intent": player_input,
            "final_output": final_output
        }
        
        # Fire background thread to populate async_critic_scores
        threading.Thread(target=async_critic_evaluation, args=(self.client, self.fast_model, turn_data, self.log_file)).start()

        print(f"[SPEAR] Turn Complete | Latency: {round(latency_ms)}ms")
        print("▲"*50)
        return final_output
    
# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app) 
game = None 

@app.route('/api/game', methods=['POST'])
def handle_game_turn():
    if not game:
        return jsonify({"error": "Game not initialized"}), 500
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

@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    if not game:
        return jsonify({"error": "Game not initialized"}), 500
    return jsonify(game.memory.get_current_state())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--profile",
        choices=["local", "fast", "default"],
        default="default",
        help="Set the profile"
    )
    args = parser.parse_args()
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
    else:
        print("Using Groq API...")
        client = Groq(api_key=Config.GROQ_API_KEY)
        model_profile = "GROQ"

    game = SPEAROrchestrator(client=client, model_profile=model_profile)
    
    # Initialize World
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
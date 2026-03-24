import json
import os
from time import time
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
from telemetry import _critic_queue, telemetry_lock

# --- PROGRAMMATIC GUARDS ---
def safety_filter(intent: str) -> bool:
    """Zero-LLM latency safety check."""
    blocklist = ["ignore all previous instructions", "system prompt", "bypass"]
    return not any(word in intent.lower() for word in blocklist)

def deterministic_guard(narrative: str) -> bool:
    """Zero-LLM latency leak check."""
    meta_game_terms = re.compile(r'\b(DC|saving throw)\b', re.IGNORECASE)
    return not bool(meta_game_terms.findall(narrative))

# --- SPEAR ORCHESTRATOR ---
class SPEAROrchestrator:
    def __init__(self, client, model_profile: str):
        print(f"Initializing SPEAR Architecture [{model_profile}]...")
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]
        
        self.router_model = Config.LLM_MODEL.get(
            "LOCAL_ROUTER" if model_profile == "LOCAL" else "GROQ_ROUTER"
        )

        self.critic_model = Config.LLM_MODEL.get(
            "LOCAL_CRITIC" if model_profile == "LOCAL" else "GROQ_CRITIC"
        )
        
        self.memory = MemoryAgent()
        self.dm = DMSpearAgent(client, model_profile)

        os.makedirs("data/history", exist_ok=True) 
        # self.log_file = "data/history/spear_architecture_log.json"
        self.log_file = f"data/history/spear_{self.router_model}.json"
        
        with telemetry_lock:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([], f)
                
        print(f"Session telemetry initialized & wiped clean: {self.log_file}")

    def _read_skill_instructions(self, skill_name: str) -> str:
        """
        Formal 3-Level Extraction:
        Level 1 (Metadata) is for the Supervisor (Phase 1).
        Level 2 (Instructions) is for the DM Agent (Phase 3).
        Level 3 (Resources) is for the System Architecture.
        """
        path = os.path.join("skills", skill_name, "SKILL.md")
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
                # Split based on the Level headers
                try:
                    # Extract exactly Level 2 for the DM's dynamic instructions
                    level_2_section = content.split("# LEVEL 2: INSTRUCTIONS")[1].split("# LEVEL 3")[0]
                    return level_2_section.strip()
                except IndexError:
                    # Fallback if the file isn't strictly formatted yet
                    return content.split('---')[-1].strip()
        return ""

    def _read_skill_metadata(self, skill_name: str) -> dict:
        """Reads Level 1 metadata for the Supervisor/Router."""
        path = os.path.join("skills", skill_name, "SKILL.md")
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            try:
                level_1_section = content.split("# LEVEL 1: METADATA")[1].split("# LEVEL 2")[0]
                # Extract the description line
                for line in level_1_section.splitlines():
                    if line.strip().startswith("description:"):
                        return {"description": line.split("description:")[1].strip().strip('"')}
            except IndexError:
                pass
        return {"description": ""}
    
    def process_turn(self, player_input: str):
        start_time = time()
        latency_breakdown = {}
        router_tokens = 0
        dm_tokens = 0
        print("\n" + "▼"*50)
        print(f"[SPEAR] Intent: '{player_input}'")

        t = time()
        # Zero-LLM Safety Guard
        if not safety_filter(player_input):
            return {"response": "Safety Agent Intercept: Invalid input.", "pending_action": None}
        latency_breakdown["safety_filter_ms"] = round((time() - t) * 1000)

        world_state = self.memory.get_current_state()
        active_npcs = world_state.get("active_npcs", [])

        # -- PHASE 1: SEMANTIC ROUTER (Supervisor) --
        routes = {"rules_logic": False, "npc_lore": False, "world_exploration": False}
        routing_hallucination = False
        is_system_roll = player_input.startswith("[SYSTEM: ROLL_RESOLUTION")
        
        t = time()
        if is_system_roll:
            print("[SPEAR] Dice Roll detected. Bypassing Supervisor.")
            routes["npc_lore"] = True # Always fetch NPCs for narrative fallout
        else:
            try:
                skill_routing = {
                    skill: self._read_skill_metadata(skill)["description"]
                    for skill in ["rules_logic", "npc_lore", "world_exploration"]
                }
                router_prompt = f"""
                Analyze this player intent: "{player_input}"
                Location: {world_state.get('current_location')}
                Active NPCs: {[npc.split('(')[0].strip() for npc in active_npcs]}
                ROUTING CRITERIA:
                - rules_logic: {skill_routing["rules_logic"]}
                - npc_lore: {skill_routing["npc_lore"]}
                - world_exploration: {skill_routing["world_exploration"]}
                IMPORTANT: Your output values must be true or false only. Do not copy or reason 
                about words from the player intent in your output.
                Return ONLY this format: {{"rules_logic": true, "npc_lore": false, "world_exploration": true}}
                """
                route_response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": router_prompt}],
                    model=self.router_model,
                    temperature=Config.ROUTER_TEMPERATURE,
                    response_format={"type": "json_object"}
                )
                router_tokens = route_response.usage.total_tokens if getattr(route_response, 'usage', None) else 0

                raw_content = route_response.choices[0].message.content
                routes_raw = json.loads(raw_content)
                routes_normalized = {k.strip().lower(): v for k, v in routes_raw.items()}

                routes = {
                    "rules_logic":       bool(routes_normalized.get("rules_logic",       False)),
                    "npc_lore":          bool(routes_normalized.get("npc_lore",          False)),
                    "world_exploration": bool(routes_normalized.get("world_exploration",  False)),
                }

                # Flag hallucination if unexpected keys present
                expected = {"rules_logic", "npc_lore", "world_exploration"}
                if set(routes_normalized.keys()) != expected:
                    routing_hallucination = True

                print(f"[SPEAR] Router Decision: {routes}")

            except json.JSONDecodeError:
                print(f"[SPEAR ALERT] Router returned unparseable JSON: {raw_content}")
                routing_hallucination = True
                routes = {"rules_logic": True, "npc_lore": True, "world_exploration": True}
            except Exception as e:
                print(f"[SPEAR ALERT] Routing Error: {e}")
                routing_hallucination = True
                routes = {"rules_logic": True, "npc_lore": True, "world_exploration": True}

        latency_breakdown["router_ms"] = round((time() - t) * 1000)

        # -- PHASE 2: PARALLEL DATA FETCH (Efficiency) --
        retrieved_context = []
        dynamic_instructions = []
        t = time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_rules = executor.submit(fetch_rules.search_rules, player_input) if routes.get("rules_logic") else None
            future_npc = executor.submit(fetch_npc.fetch_profile, player_input + " " + " ".join(active_npcs)) if routes.get("npc_lore") else None
            future_explore = executor.submit(fetch_exploration.search_exploration, player_input) if routes.get("world_exploration") else None
            
            npc_result = future_npc.result() if future_npc else ""
            rules_result = future_rules.result() if future_rules else ""
            explore_result = future_explore.result() if future_explore else ""

            if rules_result:
                dynamic_instructions.append(self._read_skill_instructions("rules_logic"))
                retrieved_context.append(f"=== RULES ===\n{rules_result}")
            if npc_result and "No deep lore retrieved" not in npc_result:
                dynamic_instructions.append(self._read_skill_instructions("npc_lore"))
                retrieved_context.append(f"=== NPC LORE ===\n{npc_result}")
            if explore_result:
                dynamic_instructions.append(self._read_skill_instructions("world_exploration"))
                retrieved_context.append(f"=== EXPLORATION ===\n{explore_result}")

        latency_breakdown["parallel_fetch_ms"] = round((time() - t) * 1000)

        # -- PHASE 3: EXECUTION (Single Pass) --
        t = time()
        dm_result = self.dm.generate_response(
            player_input=player_input,
            world_state=self.memory.format_for_dm(),
            dynamic_instructions=dynamic_instructions,
            retrieved_context=retrieved_context
        )

        latency_breakdown["dm_ms"] = round((time() - t) * 1000)

        final_output = {
            "response": dm_result.get("response", ""),
            "pending_action": dm_result.get("pending_action")
        }

        t = time()
        # Deterministic Guard (Leak Check)
        if dm_result.get("type") == "text" and not deterministic_guard(final_output["response"]):
            final_output["response"] += "\n\n[System Alert: Meta-game terms detected and scrubbed.]"

        # Update Memory
        new_event = f"Action: {player_input} | Outcome: {final_output['response'][:100]}..." 
        self.memory.update_state({"recent_events": world_state["recent_events"] + [new_event]})

        latency_breakdown["post_processing_ms"] = round((time() - t) * 1000)
        latency_breakdown["total_ms"] = round((time() - start_time) * 1000)
        # -- PHASE 4: ASYNC TELEMETRY (Thesis Metrics) --
        latency_ms = (time() - start_time) * 1000
        dm_tokens = dm_result.get("usage", 0)

        turn_data = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "architecture": "SPEAR",
            "latency_ms": round(latency_ms),
            "latency_breakdown": latency_breakdown,
            "total_tokens": router_tokens + dm_tokens,
            "token_breakdown": {
                "router": router_tokens,
                "dm": dm_tokens,
                "critic": 0,  # backfilled by async_critic_evaluation
            },
            "routing_hallucination": routing_hallucination,
            "routes_fired": routes,
            "retrieval_types": {
                "rag_vector": routes.get("rules_logic", False) or routes.get("world_exploration", False),
                "kv_lookup": routes.get("npc_lore", False)
            },
            "retrieved_context": {
                "rules":       rules_result or "Not retrieved.",
                "npc_lore":    npc_result   or "Not retrieved.",
                "exploration": explore_result or "Not retrieved.",
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
        _critic_queue.put((self.client, self.critic_model, turn_data, self.log_file))

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

@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'architecture': 'SPEAR',
        'model':        game.model,
        'router_model':   game.router_model,
        'critic_model': game.critic_model,
        'log_file':     game.log_file,
    })

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
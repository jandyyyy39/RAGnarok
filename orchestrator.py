import argparse
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import Config
from agents.safety import SafetyAgent
from agents.memory import MemoryAgent
from agents.rules_arbiter import RulesArbiter
from agents.dungeon_master import DMAgent
from agents.npc_consistency import NPCConsistencyAgent
from agents.input_classifier import InputClassifier


def transcribe_audio(audio_file_path: str) -> str:
    from groq import Groq
    client = Groq(api_key=Config.GROQ_API_KEY)
    with open(audio_file_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
            language="en",
        )
    return transcription.text


class RAGnarokOrchestrator:
    def __init__(
        self,
        use_local:     bool = False,
        use_rag:       bool = True,
        use_memory:    bool = True,
        use_npc:       bool = True,
        smart_routing: bool = True,
    ):
        self.use_rag       = use_rag
        self.use_memory    = use_memory
        self.use_npc       = use_npc
        self.smart_routing = smart_routing

        print("Initializing RAGnarok Multi-Agent System...")
        print(f"  Backend  : {'Ollama (local)' if use_local else 'Groq API'}")
        print(f"  RAG      : {'ON' if use_rag else 'OFF (ablation)'}")
        print(f"  Memory   : {'ON' if use_memory else 'OFF (ablation)'}")
        print(f"  NPC Pass : {'ON' if use_npc else 'OFF (ablation)'}")

        self.safety     = SafetyAgent()
        self.classifier = InputClassifier()
        self.memory     = MemoryAgent()          if use_memory else None
        self.arbiter    = RulesArbiter()         if use_rag    else None
        self.dm         = DMAgent(use_local=use_local)
        self.npc_agent  = NPCConsistencyAgent()  if use_npc    else None

        print("All agents online. Ready to play.\n")

    def process_turn(self, player_input: str) -> dict:
        start  = time.time()
        result = {
            "input":       player_input,
            "input_type":  None,
            "safety_pass": None,
            "ruling":      None,
            "dm_raw":      None,
            "response":    None,
            "latency_ms":  None,
            "skipped":     [],
        }

        print("\n" + "=" * 50)

        input_type = self.classifier.classify(player_input)
        result["input_type"] = input_type
        print(f"[Classifier] Input type: {input_type}")

        if not self.safety.check(player_input):
            result["safety_pass"] = False
            result["response"]    = "Safety Agent Intercept: That action violates the table's safety tools."
            result["latency_ms"]  = int((time.time() - start) * 1000)
            return result
        result["safety_pass"] = True

        world_state = ""
        if self.memory:
            print("[Memory Agent] Fetching recap...")
            world_state = self.memory.format_for_dm()
        else:
            result["skipped"].append("memory")

        ruling             = "No mechanical ruling — DM has full discretion."
        rag_skipped_reason = None
        if not self.arbiter:
            rag_skipped_reason = "ablation flag"
        elif self.smart_routing and self.classifier.should_skip_rag(input_type):
            rag_skipped_reason = f"smart routing ({input_type})"

        if rag_skipped_reason:
            print(f"[Rules Arbiter] Skipped ({rag_skipped_reason})")
            result["skipped"].append("rag")
        else:
            print("[Rules Arbiter] Consulting the SRD...")
            ruling = self.arbiter.get_ruling(player_input, world_state)
            print(f"  -> Ruling: {ruling[:100]}...")
        result["ruling"] = ruling

        print("[DM Agent] Weaving the narrative...")
        dm_response    = self.dm.generate_response(player_input, world_state, ruling)
        result["dm_raw"] = dm_response

        npc_skipped_reason = None
        if not self.npc_agent:
            npc_skipped_reason = "ablation flag"
        elif self.smart_routing and self.classifier.should_skip_npc(input_type):
            npc_skipped_reason = f"smart routing ({input_type})"

        if npc_skipped_reason:
            print(f"[NPC Agent] Skipped ({npc_skipped_reason})")
            result["skipped"].append("npc")
            final_output = dm_response
        else:
            print("[NPC Agent] Checking character sheets...")
            final_output = self.npc_agent.refine_dialogue(dm_response)

        result["response"] = final_output

        if self.memory:
            current_events = self.memory.get_current_state()["recent_events"]
            self.memory.update_state({
                "recent_events": current_events + [
                    f"Player: {player_input} | Outcome: {final_output[:100]}..."
                ]
            })

        result["latency_ms"] = int((time.time() - start) * 1000)
        print(f"[Orchestrator] Turn complete in {result['latency_ms']}ms")
        print("=" * 50 + "\n")
        return result


app  = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
game: RAGnarokOrchestrator | None = None


@app.route('/api/game', methods=['POST'])
def handle_game_turn():
    data         = request.json
    player_input = data.get('input', '').strip()
    if not player_input:
        return jsonify({'error': 'No input provided'}), 400
    turn       = game.process_turn(player_input)
    game_state = game.memory.get_current_state() if game.memory else {}
    return jsonify({'response': turn["response"], 'game_state': game_state, 'meta': turn})


@app.route('/api/transcribe', methods=['POST'])
def handle_transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio    = request.files['audio']
    tmp_path = "tmp_audio.wav"
    audio.save(tmp_path)
    try:
        text = transcribe_audio(tmp_path)
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGnarok — AI Dungeon Master")
    parser.add_argument("--local",            action="store_true")
    parser.add_argument("--no-rag",           action="store_true")
    parser.add_argument("--no-memory",        action="store_true")
    parser.add_argument("--no-npc-const",     action="store_true")
    parser.add_argument("--no-smart-routing", action="store_true")
    args = parser.parse_args()

    game = RAGnarokOrchestrator(
        use_local     = args.local,
        use_rag       = not args.no_rag,
        use_memory    = not args.no_memory,
        use_npc       = not args.no_npc_const,
        smart_routing = not args.no_smart_routing,
    )

    if game.memory:
        game.memory.update_state({
            "current_location": "The Yawning Portal Tavern",
            "active_npcs":      ["Durnan the Barkeep"],
            "recent_events":    ["The party just walked into the crowded tavern."],
        })

    app.run(port=5000, debug=True, use_reloader=False)

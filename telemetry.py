import json
import os
import queue
import threading
from config import Config

telemetry_lock = threading.Lock()
_critic_queue = queue.Queue()

def async_critic_evaluation(client, model, turn_data, log_file):
    try:
        ctx = turn_data.get("retrieved_context", {})
        rules_ctx   = ctx.get("rules",       "Not retrieved.")
        npc_ctx     = ctx.get("npc_lore",    "Not retrieved.")
        explore_ctx = ctx.get("exploration", "Not retrieved.")

        dm_response = turn_data['final_output'].get('response', '')

        # Determine response type before building the prompt
        is_tool_call = bool(turn_data['final_output'].get('pending_action'))

        if is_tool_call:
            scoring_rules = f"""
            SCORING RULES (Tool Call Turn):
            - rule_accuracy:         Did the tool call use the correct stat/skill for this action 
                                    based on the RULES context above? Score 0-10.
            - narrative_coherence:   Was calling for a dice roll the correct response to this 
                                    player intent? Score 10 if yes, 0 if no roll was needed.
            - npc_voice_consistency: No dialogue was generated. Score 10.
            """
        else:
            scoring_rules = f"""
            SCORING RULES (Narrative Turn):
            - rule_accuracy:         Does the DM response correctly reflect the RULES context above?
                                    If rules were "Not retrieved", use general 5e knowledge. Score 0-10.
            - narrative_coherence:   Did the story advance meaningfully given the player intent? Score 0-10.
            - npc_voice_consistency: 
                - No named NPCs in intent or response → inapplicable, score 10.
                - Named NPC speaks or acts → compare ONLY against NPC PROFILES above, score 0-10.
                - Named NPC present but silent → neutral, score 7.
            """

        prompt = f"""
        You are a TTRPG quality evaluator. Score this D&D 5e turn on three metrics (0-10 each).
        Use the ground truth context below as your sole reference — do not rely on outside knowledge.

        --- GROUND TRUTH CONTEXT ---
        RULES:       {rules_ctx}
        NPC PROFILES: {npc_ctx}
        EXPLORATION:  {explore_ctx}
        ---

        Player Intent: {turn_data['player_intent']}
        DM Response:   {dm_response}

        {scoring_rules}

        Output pure JSON with integer scores:
        {{"rule_accuracy": <int>, "narrative_coherence": <int>, "npc_voice_consistency": <int>, "reasoning": "<string>"}}
        """
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=Config.CRITIC_TEMPERATURE,
            response_format={"type": "json_object"}
        )
        scores = json.loads(response.choices[0].message.content)
        critic_tokens = response.usage.total_tokens if getattr(response, 'usage', None) else 0

        turn_data["async_critic_scores"] = scores
        turn_data["token_breakdown"]["critic"] = critic_tokens
        turn_data["total_tokens"] += critic_tokens

    except Exception as e:
        print(f"   [Async Critic] Failed to score turn: {e}")
        turn_data["async_critic_scores"] = {"error": str(e)}

    # Always write — even on critic failure, the turn must be logged
    finally:
        with telemetry_lock:
            history = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    history = []
            history.append(turn_data)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
        print(f"   [Async Critic] Logged: {turn_data['async_critic_scores']}")

def _critic_worker():
    while True:
        task = _critic_queue.get()
        if task is None:
            break
        client, model, turn_data, log_file = task
        async_critic_evaluation(client, model, turn_data, log_file)
        _critic_queue.task_done()

_critic_thread = threading.Thread(target=_critic_worker, daemon=True)
_critic_thread.start()
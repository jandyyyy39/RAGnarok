import json
import os
import queue
import threading

telemetry_lock = threading.Lock()
_critic_queue = queue.Queue()

def async_critic_evaluation(client, model, turn_data, log_file):
    try:
        ctx = turn_data.get("retrieved_context", {})
        rules_ctx   = ctx.get("rules",       "Not retrieved.")
        npc_ctx     = ctx.get("npc_lore",    "Not retrieved.")
        explore_ctx = ctx.get("exploration", "Not retrieved.")

        prompt = f"""
        You are a TTRPG quality evaluator. Score this turn from 0-10 on three metrics.
        You have been given the EXACT context the DM was working from — use it as ground truth.

        --- GROUND TRUTH CONTEXT ---
        RULES THE DM HAD ACCESS TO:
        {rules_ctx}

        NPC PROFILES THE DM HAD ACCESS TO:
        {npc_ctx}

        EXPLORATION RULES THE DM HAD ACCESS TO:
        {explore_ctx}
        ---

        Player Intent: {turn_data['player_intent']}
        DM Response:   {turn_data['final_output'].get('response', 'TOOL_CALL')}

        SCORING RULES:
        - rule_accuracy:         Compare DM response against RULES context.
                                 If rules were "Not retrieved", score based on general 5e knowledge.
        - narrative_coherence:   Did the story advance meaningfully given the player's intent?
        - npc_voice_consistency: 
            Step 1 - Check the Player Intent AND DM Response for any named NPCs.
            Step 2 - If NO named NPCs appear in either, this metric is inapplicable. Score 10.
            Step 3 - If an NPC IS present but speaks/acts, compare their dialogue 
                    against their profile quirks. Score based on adherence.
            Step 4 - If an NPC IS present but does NOT speak or act (e.g. they are 
                    only mentioned or react physically), score 7 as neutral — 
                    absence of dialogue is not a violation.

        Output pure JSON:
        {{"rule_accuracy": 0, "narrative_coherence": 0, "npc_voice_consistency": 0, "reasoning": "..."}}
        """
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
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
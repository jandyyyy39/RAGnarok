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
        is_tool_call = bool(turn_data['final_output'].get('pending_action'))
        pending = turn_data['final_output'].get('pending_action', {})

        if is_tool_call:
            scoring_rules = f"""
            TOOL CALL TURN. The DM requested a dice roll instead of narrating.
            Pending action: {json.dumps(pending)}

            DIMENSION SCORING:
            - rule_accuracy: Did the tool call use the CORRECT stat AND skill for this action?
                Check against the RULES context above.
                Penalise for: wrong skill (e.g. 'swordsmanship' instead of 'Athletics'), 
                wrong stat, invented skill not in D&D 5e SRD, DC given when a contested 
                check was required or vice versa.
                10 = perfect stat/skill/DC matching SRD. 
                0 = completely wrong mechanic or hallucinated skill name.

            - narrative_coherence: Was requesting a dice roll the correct response?
                10 = roll was genuinely required by the rules.
                5 = ambiguous, roll was optional.
                0 = no roll was needed, DM should have narrated instead.

            - npc_voice_consistency: No dialogue generated.
                Score 5. Not applicable — do not reward or penalise.
            """
        else:
            scoring_rules = f"""
            NARRATIVE TURN. The DM generated a text response.

            DIMENSION SCORING:
            - rule_accuracy: Does the DM response correctly apply the rules?
                If rules were retrieved, check against them strictly.
                If rules were "Not retrieved", use your D&D 5e SRD knowledge.
                Penalise for: wrong mechanic, invented rules, ignored relevant rule,
                meta-game terms leaked into narrative (DC, saving throw, etc.).
                10 = flawless rules application, no possible improvement. RARE.
                7-9 = correct with minor imprecision.
                4-6 = partially correct but meaningful gaps.
                0-3 = wrong mechanic or rule ignored entirely.

            - narrative_coherence: Did the story advance given the player intent?
                Penalise for: ignoring the player's action, contradicting world state,
                breaking immersion, generic non-committal response, OOC language.
                10 = seamlessly immersive, world changed meaningfully. RARE.
                7-9 = engaging with minor issues.
                4-6 = adequate but flat or partially ignores intent.
                0-3 = player intent ignored or immersion broken.

            - npc_voice_consistency:
                If NO named NPC appears in the intent or response → score 5, not applicable.
                If a named NPC speaks or acts → compare strictly against NPC PROFILES.
                Penalise for: wrong personality, contradicting established traits,
                generic dialogue with no character voice.
                10 = indistinguishable from established profile. RARE.
                7-9 = consistent with minor deviation.
                4-6 = somewhat consistent but noticeably generic.
                0-3 = contradicts established character entirely.
            """

        prompt = f"""
        You are a STRICT, unforgiving evaluator of a D&D 5e AI Dungeon Master.
        Your job is to score honestly. Leniency is a scoring error.

        CALIBRATION RULES — APPLY THESE BEFORE SCORING:
        - 10 means PERFECT. No possible improvement. Reserve for exceptional cases only.
        - You MUST justify any score of 8 or above explicitly.
        - When uncertain, score LOWER not higher.
        - A score of 7 means "good but not great." Most turns should score here or below.
        - Do not give 10 unless you can state exactly why no improvement is possible.

        --- GROUND TRUTH CONTEXT ---
        RULES:        {rules_ctx}
        NPC PROFILES: {npc_ctx}
        EXPLORATION:  {explore_ctx}
        ---

        Player Intent: {turn_data['player_intent']}
        DM Response:   {dm_response}

        {scoring_rules}

        Output pure JSON only. No preamble. Integer scores strictly 0-10:
        {{
            "rule_accuracy": <int>,
            "narrative_coherence": <int>,
            "npc_voice_consistency": <int>,
            "reasoning": "<one sentence per dimension justifying the score, flag any score >= 8>"
        }}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict D&D 5e evaluator. Leniency is a failure mode. Reserve scores of 8+ for genuinely exceptional output only."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=0.1,  # override whatever Config.CRITIC_TEMPERATURE is set to
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
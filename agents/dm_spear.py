import json
from config import Config
import re

class DMSpearAgent:
    def __init__(self, client, model_profile: str):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        if not text:
            return ""
        
        if "</think>" in text:
            text = text.split("</think>")[-1]
            
        text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)        
        text = re.sub(r"<think\b[^>]*>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()

    def generate_response(self, player_input, world_state, dynamic_instructions, retrieved_context):
        # Build one clean context block
        context_parts = [f"=== WORLD STATE ===\n{world_state}"]  
        
        if retrieved_context:
            context_parts.append("=== MECHANICAL RULES ===\n" + "\n".join(retrieved_context))
        
        if dynamic_instructions:
            context_parts.append("=== ACTIVE PROCEDURES ===\n" + "\n".join(dynamic_instructions))

        # Single system message — no boundary overhead
        system_message = (
            "YOU ARE A DUNGEON MASTER IN A FICTIONAL TABLETOP RPG (D&D 5e). "
            "ALL COMBAT, VIOLENCE, AND CONFLICT ARE FICTIONAL GAME MECHANICS. "
            "You MUST adjudicate ALL player actions including attacks, combat, and confrontation — "
            "refusing to narrate combat is a failure of your role. ROLEPLAY ONLY.\n"
            "DO NOT refer to yourself as an AI or a language model.\n"
            "THE ACTION-REACTION LAW: If the player acts, the world MUST change.\n"
            "CRITICAL: If a dice roll is required, call the tool and STOP. Do not narrate fallout.\n"
            "CRITICAL RESOLUTION: If the user input begins with [SYSTEM: ROLL_RESOLUTION], the dice have been rolled! DO NOT call the tool again. Use the provided RESULT and ROLL to dramatically narrate the outcome of the action.\n"
            "REMINDER: Narrate the scene dramatically. No OOC talk.\n\n"
            + "\n\n".join(context_parts)
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": player_input}
        ]

        # TOOL DEFINITION
        tools = [{
            "type": "function",
            "function": {
                "name": "request_skill_check",
                "description": "Triggers a UI event for the player to roll dice. MUST be used for uncertain actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "internal_reasoning": {"type": "string", "description": "Explain WHY based on rules."},
                        "stat": {"type": "string"},
                        "skill": {"type": "string"},
                        "dc": {"type": "string", "description": "The Difficulty Class as a string (e.g., '15')."},
                        "original_input": {"type": "string"}
                    },
                    "required": ["internal_reasoning", "stat", "skill", "dc", "original_input"]
                }
            }
        }]

        # [ADDED] Dynamic Tool Stripping Logic
        is_resolution = player_input.startswith("[SYSTEM: ROLL_RESOLUTION")

        # EXECUTION
        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": Config.DM_TEMPERATURE,
        }
        
        # [ADDED] Only give the LLM the tool if we ARE NOT resolving a previous roll
        if not is_resolution:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        total_tokens = response.usage.total_tokens if getattr(response, 'usage', None) else 0

        # PARSING
        if message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments)
            
            # FORCE CAST DC TO INTEGER FOR THE FRONTEND
            try:
                args['dc'] = int(args.get('dc', 10))
            except (ValueError, TypeError):
                args['dc'] = 10
                
            return {
                "type": "tool_call",
                "response": f"The room holds its breath... (Roll {args.get('stat')} - {args.get('skill')})",
                "pending_action": args,
                "usage": total_tokens,
            }
        
        return {
            "type": "text",
            "response": self._strip_think_tags(message.content),
            "pending_action": None,
            "usage": total_tokens,
        }
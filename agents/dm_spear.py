import json
from config import Config

class DMSpearAgent:
    def __init__(self, client, model_profile: str):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

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
                        "dc": {"type": "integer"},
                        "original_input": {"type": "string"}
                    },
                    "required": ["internal_reasoning", "stat", "skill", "dc", "original_input"]
                }
            }
        }]

        # EXECUTION
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=Config.DM_TEMPERATURE,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        total_tokens = response.usage.total_tokens if getattr(response, 'usage', None) else 0

        # PARSING
        if message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments)
            return {
                "type": "tool_call",
                "response": f"The room holds its breath... (Roll {args.get('stat')} - {args.get('skill')})",
                "pending_action": args,
                "usage": total_tokens,
            }
        
        return {
            "type": "text",
            "response": message.content,
            "pending_action": None,
            "usage": total_tokens,
        }
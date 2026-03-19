import json
from config import Config

class DMSpearAgent:
    def __init__(self, client, model_profile: str):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

    def generate_response(self, player_input: str, world_state: dict, dynamic_instructions: list, retrieved_context: list) -> dict:
        """
        Executes the narrative pass using a Persona-Preserving 'Sandwich' prompt.
        """
        
        # Hard-coded Persona
        system_instructions = (
            "SYSTEM: YOU ARE A RUTHLESS DUNGEON MASTER. ROLEPLAY ONLY.\n"
            "DO NOT refer to yourself as an AI or a language model.\n"
            "THE ACTION-REACTION LAW: If the player acts, the world MUST change.\n"
            "CRITICAL: If a dice roll is required, call the tool and STOP. Do not narrate fallout.\n"
        )
        
        # Context and Rules
        context_payload = f"WORLD STATE: {world_state}\n\n"
        if dynamic_instructions:
            context_payload += "SKILL PROCEDURES:\n" + "\n".join(dynamic_instructions) + "\n\n"
        if retrieved_context:
            context_payload += "MECHANICAL DATA:\n" + "\n".join(retrieved_context) + "\n\n"

        # Persona Reminder
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "system", "content": context_payload},
            {"role": "system", "content": "REMINDER: You are the DM. Narrate the scene dramatically. No OOC talk."},
            {"role": "user", "content": player_input}
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
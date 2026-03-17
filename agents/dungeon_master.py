import json
from config import Config

class DMAgent:
    def __init__(self, client, model_profile: str):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

    def generate_response(self, player_input: str, world_state: str, ruling: str) -> dict:
        prompt = f"""
            You are acting as the in-world Dungeon Master for an ongoing Dungeons & Dragons session.

            You receive:
            - The current WORLD STATE (facts about the scene, NPCs, and unresolved threads)
            - A RULES ARBITER RULING (how the game mechanics say this action should be resolved)
            - The PLAYER'S DECLARED ACTION

            Your job is to:
            - Respect the Arbiter's ruling and never contradict it.
            - Narrate what happens next in a vivid but concise way (2–4 short paragraphs max).
            - Keep details consistent with the WORLD STATE.
            - Emphasize consequences and new hooks (what changed, what is now at stake, what can be done next).
            - Favor second person ("you") addressing the player.

            WORLD STATE:
            {world_state}

            RULES ARBITER RULING:
            {ruling}

            PLAYER ACTION:
            "{player_input}"

            INSTRUCTIONS:
            1. If the Arbiter's ruling says a dice roll is required (e.g., a skill check or saving throw), DO NOT decide or describe the result. Use the 'request_skill_check' tool immediately instead of narrating the outcome.
            2. If no roll is required, describe the scene and the outcome based on the ruling. Make sure the narration:
            - Clearly reflects success, failure, or partial success as implied by the ruling.
            - Updates the fiction in a way that will still make sense later.
            - Ends with either an implicit or explicit prompt for what the player might do next.
            3. Do not invent new mechanical rulings yourself; rely only on the Arbiter's ruling and the existing world state.
        """
        
        # 1. Define the Tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "request_skill_check",
                    "description": "Triggers a UI event for the player to roll dice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stat": {"type": "string", "description": "e.g., Charisma, Strength"},
                            "skill": {"type": "string", "description": "e.g., Intimidation, Athletics"},
                            "dc": {"type": "integer", "description": "The Difficulty Class (e.g., 18)"},
                            "dice_type": {"type": "string", "description": "Usually 'd20'"},
                            "original_input": {"type": "string", "description": "The player's original request that triggered this roll."}
                        },
                        "required": ["stat", "skill", "dc", "dice_type", "original_input"]
                    }
                }
            }
        ]

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Dungeon Master for a Dungeons & Dragons game. "
                        "You are fair, respect player agency, and keep the tone fast-paced and entertaining. "
                        "You may have a ruthless, Monty-Python-esque flair, but never at the expense of clarity "
                        "or consistency with the rules arbiter's ruling and the established world state. "
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=Config.DM_TEMPERATURE,
            tools=tools,
            tool_choice="auto",
        )
        
        message = response.choices[0].message
        
        # 2. Check if the model decided to call the tool
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            return {
                "type": "tool_call", 
                "action": args,
                "narrative": f"The room holds its breath... (Roll {args['stat']} - {args['skill']})"
            }
        
        # 3. Otherwise, return normal text
        return {
            "type": "text", 
            "narrative": message.content
        }
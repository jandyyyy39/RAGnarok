import json
from config import Config

class DMAgent:
    def __init__(self, client, model_profile: str):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

    def generate_response(self, player_input: str, world_state: str, ruling: str) -> dict:
        prompt = f"""
        You are an expert Dungeon Master. Respond to the player's action.
        
        {world_state}
        
        RULES ARBITER RULING:
        {ruling}
        
        PLAYER ACTION:
        "{player_input}"
        
        INSTRUCTIONS:
        1. If the Arbiter's ruling requires a dice roll, DO NOT describe the outcome. Use the 'request_skill_check' tool immediately.
        2. If no roll is required, describe the scene and the outcome. Keep it immersive and engaging.
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
                {"role": "system", "content": "You are a ruthless, Monty-Python-esque Dungeon Master."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=Config.DM_TEMPERATURE,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        total_tokens = response.usage.total_tokens if getattr(response, 'usage', None) else 0
        
        # 2. Check if the model decided to call the tool
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            return {
                "type": "tool_call", 
                "action": args,
                "narrative": f"The room holds its breath... (Roll {args['stat']} - {args['skill']})",
                "usage": total_tokens,
            }
        
        # 3. Otherwise, return normal text
        return {
            "type": "text", 
            "narrative": message.content,
            "usage": total_tokens,
        }
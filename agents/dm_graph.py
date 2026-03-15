import json
from config import Config

class DMGraphAgent:
    def __init__(self, client, model_profile: str):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

    def generate_response(self, system_prompt: str, player_input: str, world_state: str, context: str, critic_feedback: str = "") -> dict:
        """
        Generates a response using Progressive Disclosure (dynamic system prompt)
        and supports Cyclic Reflection (critic feedback).
        """
        
        # Build the dynamic user prompt
        user_prompt = f"""
        {world_state}
        
        RETRIEVED CONTEXT / RULES:
        {context}
        
        PLAYER ACTION:
        "{player_input}"
        """

        # Inject the Critic's scolding if the DM failed a previous attempt
        if critic_feedback:
            user_prompt += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS DRAFT:\n{critic_feedback}\nYou MUST correct this in your new response."

        # Keep the exact same tool definition as baseline
        # tools = [
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "request_skill_check",
        #             "description": "Triggers a UI event for the player to roll dice.",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "stat": {"type": "string", "description": "e.g., Charisma, Strength"},
        #                     "skill": {"type": "string", "description": "e.g., Intimidation, Athletics"},
        #                     "dc": {"type": "integer", "description": "The Difficulty Class (e.g., 18)"},
        #                     "dice_type": {"type": "string", "description": "Usually 'd20'"},
        #                     "original_input": {"type": "string", "description": "The player's original request that triggered this roll."}
        #                 },
        #                 "required": ["stat", "skill", "dc", "dice_type", "original_input"]
        #             }
        #         }
        #     }
        # ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "request_skill_check",
                    "description": "Triggers a UI event for the player to roll dice. MUST be used if the player is attacking, persuading, sneaking, or intimidating.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "internal_reasoning": {"type": "string", "description": "Explain WHY a dice roll is or isn't needed based on the rules before setting the other fields."},
                            "stat": {"type": "string", "description": "e.g., Charisma, Strength"},
                            "skill": {"type": "string", "description": "e.g., Intimidation, Athletics, or 'Attack Roll'"},
                            "dc": {"type": "integer", "description": "The Difficulty Class (e.g., 18)"},
                            "dice_type": {"type": "string", "description": "Usually 'd20'"},
                            "original_input": {"type": "string", "description": "The player's original request that triggered this roll."}
                        },
                        "required": ["internal_reasoning", "stat", "skill", "dc", "dice_type", "original_input"]
                    }
                }
            }
        ]

        # Execute
        # base_rules = (
        #     "You are a Dungeon Master for a tabletop roleplaying game. "
        #     "Simulated fantasy violence and conflict are permitted and expected. "
        #     "RULE 1: NEVER write dialogue for the player character. "
        #     "RULE 2: NEVER dictate the player's actions, feelings, or thoughts beyond what they explicitly stated. "
        #     "RULE 3: Always end your turn by describing the environment, the NPC's reaction, or the immediate consequence, leaving the next move entirely to the player."
        # )
        base_rules = (
            "You are a Dungeon Master for a tabletop roleplaying game. "
            "Simulated fantasy violence and conflict are permitted and expected. "
            "RULE 1: NEVER write dialogue for the player character. "
            "RULE 2: NEVER dictate the player's actions, feelings, or thoughts beyond what they explicitly stated. "
            "RULE 3: Always end your turn by leaving the next move entirely to the player. "
            "RULE 4: THE ACTION-REACTION LAW. The world is alive and proactive. If the player takes a significant action, the world MUST react with a concrete, state-changing action. "
            "CRITICAL EXCEPTION TO RULE 4: If the player's action triggers a rule requiring a dice roll (e.g., attacking, intimidating, persuading), using the `request_skill_check` tool IS the required action. You MUST call the tool and you MUST NOT narrate the NPC's reaction until the roll is resolved. DO NOT narrate weapon stats or damage dice directly in the story text."
        )
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"{base_rules}\n\n{system_prompt}"},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model,
            temperature=Config.DM_TEMPERATURE,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # Parse the output
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            return {
                "type": "tool_call", 
                "action": args,
                "narrative": f"The room holds its breath... (Roll {args['stat']} - {args['skill']})"
            }
        
        return {
            "type": "text", 
            "narrative": message.content
        }
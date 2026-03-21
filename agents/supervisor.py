import json
from config import Config

class SupervisorAgent:
    def __init__(self, client, model_profile: str):
        self.client = client
        self.model = Config.LLM_MODEL[model_profile]

    def route_intent(self, user_input: str, skills_metadata: dict) -> str:
        """
        Analyzes the player's input against Level 1 Skill Metadata and routes to the correct skill.
        """
        # 1. Format the Level 1 Metadata for the prompt
        skills_list = ""
        for skill_name, description in skills_metadata.items():
            skills_list += f"- **{skill_name}**: {description}\n"
            
        # 2. The Strict System Prompt
        system_prompt = """You are the Semantic Router for an advanced multi-agent system. 
Your ONLY job is to analyze the user's input and select the single most appropriate Agent Skill from the available list.
Do not answer the user. Do not narrate. 

You MUST respond in strict JSON format matching this exact schema:
{
    "reasoning": "A 1-sentence explanation of why this skill fits.",
    "selected_skill": "The exact name of the skill."
}"""

        user_prompt = f"""
AVAILABLE SKILLS:
{skills_list}

PLAYER INPUT:
"{user_input}"

Route this input to the correct skill.
"""
        
        try:
            print("--- SUPERVISOR: Analyzing Intent ---")
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.1, # Ice cold. No hallucinations allowed.
                response_format={"type": "json_object"} # Forces strict JSON parsing
            )
            
            # 3. Parse the output
            result = json.loads(response.choices[0].message.content)
            selected_skill = result.get("selected_skill")
            
            print(f"> Reasoning: {result.get('reasoning')}")
            print(f"> Selected: {selected_skill}")
            
            # 4. The Safety Net: Did the LLM hallucinate a skill that doesn't exist?
            if selected_skill in skills_metadata:
                return selected_skill
            else:
                print(f"Supervisor Hallucinated Skill: '{selected_skill}'. Defaulting to social_lore.")
                return "social_lore"
                
        except Exception as e:
            print(f"Supervisor API Error: {e}. Defaulting to social_lore.")
            return "social_lore"
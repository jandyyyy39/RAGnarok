from groq import Groq
from openai import OpenAI
from config import Config


class DMAgent:
    def __init__(self, use_local: bool = False):
        if use_local:
            self.client = OpenAI(
                base_url = Config.OLLAMA_BASE_URL,
                api_key  = "ollama",
            )
            self.model = Config.LLM_MODELS["local"]
            print("[DM Agent] Using LOCAL Ollama model:", self.model)
        else:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.model  = Config.LLM_MODEL
            print("[DM Agent] Using GROQ model:", self.model)

    def generate_response(self, player_input: str, world_state: str, ruling: str) -> str:
        prompt = f"""
        You are an expert Dungeon Master. Respond to the player's action.

        {world_state}

        RULES ARBITER RULING:
        {ruling}

        PLAYER ACTION:
        "{player_input}"

        INSTRUCTIONS:
        1. Describe the scene and the outcome of the player's action.
        2. Incorporate the mechanical requirements from the Rules Arbiter.
        3. Keep the tone immersive and engaging.
        """
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a master storyteller and Dungeon Master."},
                {"role": "user",   "content": prompt}
            ],
            model       = self.model,
            temperature = Config.DM_TEMPERATURE,
        )
        return response.choices[0].message.content

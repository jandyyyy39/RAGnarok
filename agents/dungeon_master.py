from groq import Groq
<<<<<<< Updated upstream
from config import Config

class DMAgent:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        # Eventually, swap this with fine-tuned FIREBALL model via vLLM
        self.model = Config.LLM_MODEL 

=======
from openai import OpenAI
from config import Config


class DMAgent:
    def __init__(self, use_local: bool = False):
        if use_local:
            # Ollama exposes an OpenAI-compatible endpoint — same call syntax, local weights
            self.client = OpenAI(
                base_url=Config.OLLAMA_BASE_URL,
                api_key="ollama",           # Ollama doesn't require a real key
            )
            self.model = Config.LLM_MODELS["local"]   # e.g. "ragnarok-dm" fine-tuned GGUF
            print("[DM Agent] Using LOCAL Ollama model:", self.model)
        else:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.model  = Config.LLM_MODEL              # llama-3.3-70b-versatile via Groq
            print("[DM Agent] Using GROQ model:", self.model)

>>>>>>> Stashed changes
    def generate_response(self, player_input: str, world_state: str, ruling: str) -> str:
        prompt = f"""
        You are an expert Dungeon Master. Respond to the player's action.

        {world_state}

        RULES ARBITER RULING:
        {ruling}

        PLAYER ACTION:
        "{player_input}"
<<<<<<< Updated upstream
        
=======

>>>>>>> Stashed changes
        INSTRUCTIONS:
        1. Describe the scene and the outcome of the player's action.
        2. Incorporate the mechanical requirements from the Rules Arbiter.
        3. Keep the tone immersive and engaging.
        """
        response = self.client.chat.completions.create(
            messages=[
<<<<<<< Updated upstream
                {"role": "system", "content": "You are a master storyteller."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=Config.DM_TEMPERATURE
        )
        return response.choices[0].message.content
=======
                {"role": "system", "content": "You are a master storyteller and Dungeon Master."},
                {"role": "user",   "content": prompt}
            ],
            model=self.model,
            temperature=Config.DM_TEMPERATURE,
        )
        return response.choices[0].message.content
>>>>>>> Stashed changes

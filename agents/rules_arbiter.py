from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config


class RulesArbiter:
    def __init__(self):
        self.client      = Groq(api_key=Config.GROQ_API_KEY)
        self.model       = Config.LLM_MODEL['GROQ']
        self.embeddings  = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory  = str(Config.CHROMA_DIR),
            embedding_function = self.embeddings,
        )

    def get_ruling(self, player_action: str, world_context: str) -> str:
        relevant_rules = self.vectorstore.similarity_search(player_action, k=3)
        context_text   = "\n".join([doc.page_content for doc in relevant_rules])

        prompt = f"""
        SYSTEM: You are the Rules Arbiter for a D&D 5e game.
        Your job is to interpret the player's intent based on the official rules.

        WORLD CONTEXT: {world_context}
        PLAYER ACTION: "{player_action}"

        RELEVANT RULES FROM SRD:
        {context_text}

        INSTRUCTIONS:
        1. Determine if the action is possible.
        2. If possible, specify the required Ability Check, Saving Throw, or Attack Roll.
        3. Set a Difficulty Class (DC) based on the world context and rules.
        4. Be concise. Do not describe the narrative outcome; only provide the mechanical ruling.
        """

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise D&D 5e rules engine."},
                {"role": "user",   "content": prompt}
            ],
            model       = self.model,
            temperature = Config.ARBITER_TEMPERATURE,
        )
        return response.choices[0].message.content

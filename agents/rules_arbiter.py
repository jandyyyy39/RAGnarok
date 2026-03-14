<<<<<<< Updated upstream
import os
from dotenv import load_dotenv
=======
>>>>>>> Stashed changes
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class RulesArbiter:
<<<<<<< Updated upstream
    def __init__(self, db_path="chroma_db"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.model = "llama-3.3-70b-versatile"
=======
    def __init__(self):
        self.client     = Groq(api_key=Config.GROQ_API_KEY)
        self.model      = Config.LLM_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=str(Config.CHROMA_DIR),
            embedding_function=self.embeddings,
        )
>>>>>>> Stashed changes

    def get_ruling(self, player_action: str, world_context: str) -> str:
        """
        Retrieves relevant SRD rules and returns a concise mechanical ruling
        (ability check type, DC, roll required) — no narrative output.
        """
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
            model=self.model,
<<<<<<< Updated upstream
            temperature=0.1, # Low temperature for consistency
=======
            temperature=Config.ARBITER_TEMPERATURE,
>>>>>>> Stashed changes
        )
        return response.choices[0].message.content

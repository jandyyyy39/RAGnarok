import os
import torch
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config

load_dotenv()


class RulesArbiter:
    """
    Main-branch retrieval/ruling behavior with backward-compatible constructor and return type.
    """
    def __init__(self, client=None, model_profile: str = "GROQ", db_path=None):
        self.client = client or Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.LLM_MODEL.get(model_profile, Config.LLM_MODEL["GROQ"])
        db_path = db_path or str(Config.CHROMA_DIR)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Rules Arbiter] Using {device.upper()} for sentence embeddings.")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": device},
        )
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

    def get_ruling(self, player_action: str, world_context: str) -> str:
        """
        Determines if an action is valid and what rolls are required.
        """
        # Retrieve the most relevant rules from the RAG database
        relevant_rules = self.vectorstore.similarity_search(player_action, k=3)
        context_text = "\n".join([doc.page_content for doc in relevant_rules])

        # Construct the prompt for the Arbiter
        prompt = f"""
        SYSTEM: You are the Rules Arbiter for a D&D 5e game. 
        Your job is to interpret the player's intent based on the official rules.
        
        WORLD CONTEXT: {world_context}
        PLAYER ACTION: "{player_action}"
        
        RELEVANT RULES FROM SRD:
        {context_text}
        
        INSTRUCTIONS:
        1. Determine if the immediate action is possible.
        2. Identify EXACTLY ONE primary Ability Check, Saving Throw, or Attack Roll required to attempt the action.
        3. Do NOT provide secondary checks, conditional logic, or "social fallout" rolls. Pick the single most important check.
        4. Set a single Difficulty Class (DC).
        5. Output strict, concise mechanics only. No conversational text.
        """

        # Get the mechanical decision from the LLM
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise D&D 5e rules engine."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.1, # Low temperature for consistency
            max_tokens=150,
        )
        
        return response.choices[0].message.content

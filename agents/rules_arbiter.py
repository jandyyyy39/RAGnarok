import os
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

class RulesArbiter:
    def __init__(self, db_path="chroma_db"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.model = "llama-3.3-70b-versatile"

    def get_ruling(self, player_action: str, world_context: str):
        """
        Determines if an action is valid and what rolls are required.
        """
        # 1. Retrieve the most relevant rules from the RAG database
        relevant_rules = self.vectorstore.similarity_search(player_action, k=3)
        context_text = "\n".join([doc.page_content for doc in relevant_rules])

        # 2. Construct the prompt for the Arbiter
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

        # 3. Get the mechanical decision from the LLM
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise D&D 5e rules engine."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.1, # Low temperature for consistency
        )

        return response.choices[0].message.content

# Quick Test
if __name__ == "__main__":
    arbiter = RulesArbiter()
    # Mocking a scenario where a player tries to do something complex
    test_context = "The party is in a crumbling stone tower. Rain is making the floors slick."
    test_action = "I want to run up the wall, backflip over the guard, and stab him in the neck."
    
    ruling = arbiter.get_ruling(test_action, test_context)
    print(f"RULE ARBITER RULING:\n{ruling}")
import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

# Load the API key
load_dotenv()
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define the strict JSON schema using Pydantic
class NPCProfile(BaseModel):
    name: str = Field(description="The NPC's full name")
    race_and_class: str = Field(description="e.g., Dwarf Blacksmith, Elf Rogue")
    current_location: str = Field(description="Where the players will find them")
    core_motivation: str = Field(description="What drives them? e.g., 'Wants to avenge their brother'")
    hidden_secret: str = Field(description="Something they are hiding from the players")
    knowledge_state: str = Field(description="What they currently know about the world state or the players")
    dialogue_quirks: str = Field(description="How they speak. e.g., 'Stutters when lying', 'Uses overly formal words'")

class NPCList(BaseModel):
    npcs: list[NPCProfile]

def generate_npcs():
    print("Summoning NPCs from the void...")
    
    prompt = prompt = """
    You are a master Dungeon Master creating deep, complex NPCs for a D&D 5e campaign. 
    Generate a list of 20 unique, interesting NPCs. Give them secrets, varied motivations, 
    and specific dialogue quirks. Ensure a mix of friendly, hostile, and neutral characters.
    
    You MUST return the data in a strict JSON format that exactly matches this structure for each NPC:
    {
        "npcs": [
            {
                "name": "Eldrin the Wise",
                "race_and_class": "Elf Wizard",
                "current_location": "The Prancing Pony Tavern",
                "core_motivation": "Seeking the lost amulet of his ancestors.",
                "hidden_secret": "He actually stole the amulet himself years ago.",
                "knowledge_state": "Knows the players are looking for a guide.",
                "dialogue_quirks": "Constantly strokes his beard and speaks in metaphors."
            }
        ]
    }
    Do not add any other keys like 'attitude' or 'alignment'. Stick STRICTLY to the keys above.
    """

    # Call the Groq API, forcing it to return our exact JSON schema
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful JSON data generator."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile", # Using Llama 3 70B for high-quality creative writing
        temperature=0.7,
        # Guarantees perfect JSON parsing
        response_format={"type": "json_object"}, 
    )

    # Parse and save
    response_content = chat_completion.choices[0].message.content
    
    # Load into Pydantic model to ensure it is 100% valid
    parsed_data = NPCList.model_validate_json(response_content)
    
    with open("data/npc_profiles.json", "w") as f:
        json.dump(parsed_data.model_dump(), f, indent=4)
        
    print(f"Successfully generated {len(parsed_data.npcs)} NPCs and saved to data/npc_profiles.json!")

if __name__ == "__main__":
    # Create data dir if it doesn't exist
    os.makedirs("data", exist_ok=True)
    generate_npcs()
from config import Config
from agents.safety import SafetyAgent
from agents.memory import MemoryAgent
from agents.rules_arbiter import RulesArbiter
from agents.dungeon_master import DMAgent
from agents.npc_consistency import NPCConsistencyAgent

class RAGnarokOrchestrator:
    def __init__(self):
        print("Initializing RAGnarok Multi-Agent System...")
        self.safety = SafetyAgent()
        self.memory = MemoryAgent()
        self.arbiter = RulesArbiter()
        self.dm = DMAgent()
        self.npc_agent = NPCConsistencyAgent()
        print("All agents online. Ready to play.\n")

    def process_turn(self, player_input: str):
        print("\n" + "="*50)
        
        # 1. Safety Check
        if not self.safety.check(player_input):
            return "Safety Agent Intercept: That action violates the table's safety tools."

        # 2. Retrieve World State
        print("[Memory Agent] Fetching recap...")
        world_state = self.memory.format_for_dm()

        # 3. Rules Arbiter Assessment
        print("[Rules Arbiter] Consulting the SRD...")
        ruling = self.arbiter.get_ruling(player_input, world_state)
        print(f"  -> Arbiter says: {ruling[:100]}...")

        # 4. Narrative Generation
        print("[DM Agent] Weaving the narrative...")
        dm_response = self.dm.generate_response(player_input, world_state, ruling)

        # 5. NPC Consistency Pass
        print("[NPC Agent] Checking character sheets...")
        final_output = self.npc_agent.refine_dialogue(dm_response)

        # 6. Update Memory
        current_events = self.memory.get_current_state()["recent_events"]
        new_event = f"Player: {player_input} | Outcome: {final_output[:100]}..." # Truncated to save token space
        
        self.memory.update_state({
            "recent_events": current_events + [new_event]
        })

        print("="*50 + "\n")
        return final_output

if __name__ == "__main__":
    game = RAGnarokOrchestrator()
    
    # Initialize starting scenario
    game.memory.update_state({
        "current_location": "The Yawning Portal Tavern",
        "active_npcs": ["Durnan the Barkeep"],
        "recent_events": ["The party just walked into the crowded tavern."]
    })

    print("Welcome to RAGnarok. Type 'quit' to exit.")
    print("DM: You step into the bustling Yawning Portal tavern. Durnan wipes a glass behind the bar. What do you do?")
    
    while True:
        user_input = input("\nPlayer: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        response = game.process_turn(user_input)
        print(f"\nDM: {response}")
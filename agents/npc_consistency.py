import json
import os
from config import Config

class NPCConsistencyAgent:
    def __init__(self, client, model_profile: str, npc_db_path="data/npc_profiles.json"):
        self.npc_db_path = npc_db_path
        self.client = client
        self.model = getattr(Config, model_profile)['LLM_MODEL']
        
        # Load the database into memory immediately on startup
        self.npcs = self._load_npcs()

    def _load_npcs(self):
        """Loads the JSON file and creates a fast O(1) lookup dictionary."""
        if not os.path.exists(self.npc_db_path):
            print(f"[NPC Agent] WARNING: {self.npc_db_path} not found. I cannot enforce consistency without data.")
            return {}
        
        with open(self.npc_db_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Map the NPC name directly to its profile for easy access
                return {npc['name']: npc for npc in data.get('npcs', [])}
            except json.JSONDecodeError:
                print(f"[NPC Agent] ERROR: {self.npc_db_path} is corrupted JSON. Fix your data.")
                return {}

    def refine_dialogue(self, dm_output: str) -> str:
        """
        Intercepts DM output, detects active NPCs, and rewrites dialogue to match their profiles.
        """
        # Detect which NPCs are actually in the current narrative
        active_npcs = []
        for name, profile in self.npcs.items():
            # Check if their full name or just their first name is in the DM's text
            first_name = name.split()[0]
            if name in dm_output or first_name in dm_output:
                active_npcs.append(profile)

        # If no known NPCs are present, don't waste an API call. Pass it through.
        if not active_npcs:
            return dm_output

        # If we found NPCs, we force Groq to rewrite the scene.
        print(f"[NPC Agent] Detected {len(active_npcs)} known NPC(s). Enforcing personality quirks...")
        profiles_text = json.dumps(active_npcs, indent=2)
        
        prompt = f"""
        You are the NPC Consistency Enforcer. 
        The Dungeon Master just generated the following narrative, but the NPCs might sound generic.
        
        ORIGINAL NARRATIVE:
        {dm_output}
        
        RELEVANT NPC PROFILES:
        {profiles_text}
        
        YOUR JOB:
        Rewrite the ORIGINAL NARRATIVE so that the dialogue and actions of the mentioned NPCs strictly match their 'dialogue_quirks', 'core_motivation', and 'hidden_secret'.
        - Do NOT change the mechanical outcome of the narrative.
        - Do NOT add new events or change what happens.
        - ONLY flavor the dialogue and descriptions of the NPCs.
        - Output ONLY the revised narrative. Do not include any introductory or concluding remarks.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise narrative editor. You only output the final revised text."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.6 # Slightly lower temperature for editing rather than pure generation
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"[NPC Agent] Groq rewrite failed: {e}. Falling back to original DM output.")
            return dm_output
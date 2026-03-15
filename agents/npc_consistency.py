import json
import os
from groq import Groq
from config import Config


class NPCConsistencyAgent:
    def __init__(self, client=None, npc_db_path=None):
        self.client      = client or Groq(api_key=Config.GROQ_API_KEY)
        self.model       = Config.LLM_MODEL['GROQ']
        self.npc_db_path = npc_db_path or str(Config.NPC_DB_PATH)
        self.npcs        = self._load_npcs()

    def _load_npcs(self):
        if not os.path.exists(self.npc_db_path):
            print(f"[NPC Agent] WARNING: {self.npc_db_path} not found. Cannot enforce consistency.")
            return {}
        with open(self.npc_db_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return {npc['name']: npc for npc in data.get('npcs', [])}
            except json.JSONDecodeError:
                print(f"[NPC Agent] ERROR: {self.npc_db_path} is corrupted JSON.")
                return {}

    def refine_dialogue(self, dm_output: str) -> str:
        active_npcs = [
            profile for name, profile in self.npcs.items()
            if name in dm_output or name.split()[0] in dm_output
        ]

        if not active_npcs:
            return dm_output

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
        Rewrite the ORIGINAL NARRATIVE so that the dialogue and actions of the mentioned NPCs
        strictly match their 'dialogue_quirks', 'core_motivation', and 'hidden_secret'.
        - Do NOT change the mechanical outcome of the narrative.
        - Do NOT add new events or change what happens.
        - ONLY flavor the dialogue and descriptions of the NPCs.
        - Output ONLY the revised narrative. No introductory or concluding remarks.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise narrative editor. Output only the final revised text."},
                    {"role": "user",   "content": prompt}
                ],
                model       = self.model,
                temperature = 0.6,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[NPC Agent] Rewrite failed: {e}. Falling back to original DM output.")
            return dm_output

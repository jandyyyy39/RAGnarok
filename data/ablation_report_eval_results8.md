# Ablation Study Report (eval_results8)

**Baseline:** Full Pipeline (Groq)

## 1) Quantitative evaluation

Table summarizes mean metrics across the evaluated inputs (typically 8 for the final run).

| Experiment | ROUGE-L | BERTScore | Rule Coverage | chrF | Latency (ms) | LLM Composite | Safety Pass Rate | Expected Groq Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bare DM Only | 0.048 | 0.793 | 0.065 | 0.110 | 2247 | 4.375 | 1.000 | 2 |
| Full Pipeline (Groq) | 0.042 | 0.792 | 0.272 | 0.108 | 9770 | 4.750 | 1.000 | 4 |
| Local FT (No RAG) | 0.066 | 0.785 | 0.022 | 0.104 | 206089 | 2.031 | 1.000 | 2 |
| Local FT (RAG On) | 0.068 | 0.800 | 0.022 | 0.123 | 72939 | 1.531 | 1.000 | 3 |
| No Memory | 0.051 | 0.796 | 0.261 | 0.113 | 3296 | 4.719 | 1.000 | 4 |
| No NPC Pass | 0.045 | 0.795 | 0.239 | 0.109 | 8079 | 4.688 | 1.000 | 3 |
| No RAG | 0.046 | 0.794 | 0.082 | 0.105 | 1649 | 4.594 | 1.000 | 3 |

### Cost / latency proxy

True monetary cost requires provider-specific token pricing and token counts. For reproducibility, we use latency plus an *expected Groq call count* derived from which agents are enabled (Rules Arbiter, DM Agent, NPC Consistency, and LLM-as-judge). This gives a stable, comparative proxy for cost.

## 2) Qualitative analysis (example runs)

### Bare DM Only

- **Best example**: ROUGE-L=0.0 | LLM Composite=4.75
  - Input: *seeing as I am still stealthed, draws out shortbow*
  - Output: As you remain shrouded in the shadows, your stealthy form undetected by the unaware guards, you slowly draw out your shortbow. The gentle creak of the bow's wooden limbs and the soft whisper of the string as it's pulled taut are the only sounds that betray your movement. Your heart beats in anticipation, your senses heightened as you prepare for the potential chaos that may ensue.  The air is heavy with tension, the flickering torches casting eerie silhouettes on the stone walls as the guards continue their mundane...
- **Worst example**: ROUGE-L=0.0267 | LLM Composite=3.75
  - Input: !cast mass -t kei -t Ed -t Ceph -t Lav
  - Output: As the words leave your lips, the air is charged with anticipation. The dimly lit chamber, with its intricate network of ancient ruins, seems to hold its breath in expectation. Your companions, Kei, Ed, Ceph, and Lav, stand ready, their eyes fixed on you with a mix of curiosity and trepidation.  With a sweep of your arm, you begin the incantation, feeling the familiar surge of magical energy coursing through your veins. The words of the spell, a complex series of syllables and gestures, flow from your lips like a r...

### Full Pipeline (Groq)

- **Best example**: ROUGE-L=0.0864 | LLM Composite=4.75
  - Input: ) *The basilisk moves forward, now only with 6 targets to chose from* *Attack of Opportunity from Polearm Master as it moves forward.*
  - Output: As the basilisk takes a deliberate step forward, its unblinking gaze sweeps across the remaining six targets, its body language exuding an aura of calculated malevolence. The air is heavy with tension, the patrons of the Yawning Portal Tavern frozen in a mix of fear and awe. The moment the basilisk moves, you seize the opportunity, your Polearm Master training kicking in as you swiftly position yourself for an Attack of Opportunity.  With a swift and practiced motion, you bring your polearm to bear, aiming for a vu...
- **Worst example**: ROUGE-L=0.0864 | LLM Composite=4.75
  - Input: ) *The basilisk moves forward, now only with 6 targets to chose from* *Attack of Opportunity from Polearm Master as it moves forward.*
  - Output: As the basilisk takes a deliberate step forward, its unblinking gaze sweeps across the remaining six targets, its body language exuding an aura of calculated malevolence. The air is heavy with tension, the patrons of the Yawning Portal Tavern frozen in a mix of fear and awe. The moment the basilisk moves, you seize the opportunity, your Polearm Master training kicking in as you swiftly position yourself for an Attack of Opportunity.  With a swift and practiced motion, you bring your polearm to bear, aiming for a vu...

### Local FT (No RAG)

- **Best example**: ROUGE-L=0.156 | LLM Composite=3.25
  - Input: ) *The basilisk moves forward, now only with 6 targets to chose from* *Attack of Opportunity from Polearm Master as it moves forward.*
  - Output: *The beast seems to only charge ahead, now with only 5 targets to choose from... it's a good thing indeed, as the creature charges forward toward the end of the hall, before eventually becoming a part of the hallway itself....* *This one was also right up next to Karmag as well, the smell of fresh cooked meat being strong but not quite satisfying.*
- **Worst example**: ROUGE-L=0.022 | LLM Composite=1.0
  - Input: *The violet magic came over her again, creating another shield to cover her arms, now covered in blade marks and gun shot wounds. She lowered her guard and stood there in front of her, the magic still coursing violently ...
  - Output: *The creature, now only with 6 targets to choose from... it's a good thing i... Player: *sees as thi...                           RULES ARBITER RULING:         No mechanical ruling — DM has full discretion.          PLAYER ACTION:         "*The violet magic came over her again, creating another shield to cover her arms, now covered in blade marks and gun shot wounds. She lowers her guard and stands there in front of her, the magic still coursing violently through her. Her voice is gruff, but calm and genuine*  "......

### Local FT (RAG On)

- **Best example**: ROUGE-L=0.1212 | LLM Composite=2.0
  - Input: ) *The basilisk moves forward, now only with 6 targets to chose from* *Attack of Opportunity from Polearm Master as it moves forward.*
  - Output: *And then, cuven was ready to fire!* *it was a bit more difficult to aim with such numbers but he was going to make some distance as it was already casted out in time!** "Fuck fuck.. I need to find something else.." ```The spell was casted out! As it was already time to move along.```
- **Worst example**: ROUGE-L=0.0 | LLM Composite=1.0
  - Input: *seeing as I am still stealthed, draws out shortbow*
  - Output: ||"Iam occure, cuven"|| **Giant**  ||"Iam occure, cuven"||  **Giant**  ||"Iam occure, cuven"||  **Giant||**  ```The hells were going on. As it was a bit to show more of the haregon. Seemed that some was making distance to . And moving up the rabbit who could only ever respond with something else as they'd be able to make more distance!```

### No Memory

- **Best example**: ROUGE-L=0.1337 | LLM Composite=4.75
  - Input: ) *The basilisk moves forward, now only with 6 targets to chose from* *Attack of Opportunity from Polearm Master as it moves forward.*
  - Output: As the basilisk lumbers forward, its scaly body undulating with each step, the air seems to vibrate with malevolent energy. The very ground appears to tremble beneath its feet, as if the earth itself is wary of the creature's deadly gaze. With a swift motion, you seize the initiative, leveraging your mastery of the polearm to strike at the basilisk as it moves within range. The opportunity attack is a fleeting window of advantage, and you aim to capitalize on it.  With a swift and practiced motion, you thrust the p...
- **Worst example**: ROUGE-L=0.0732 | LLM Composite=4.5
  - Input: *Immediately stabs 8 times, not even bothering to stop when the mark appeared on the third one*
  - Output: As you lunge forward, your blade dances in the dimly lit chamber, striking true eight times in rapid succession. The air is filled with the sound of clashing steel and the soft, unsettling whisper of the cloaker's presence. Your attacks are a blur of motion, each strike aimed at the cloaker with precision and deadly intent. The mark, a glowing sigil of fate, appears on the third strike, but you press on, undeterred by its ominous glow.  The first few strikes bite deep into the cloaker's dark, membranous flesh, caus...

### No NPC Pass

- **Best example**: ROUGE-L=0.1212 | LLM Composite=4.75
  - Input: ) *The basilisk moves forward, now only with 6 targets to chose from* *Attack of Opportunity from Polearm Master as it moves forward.*
  - Output: As the basilisk moves forward, its eyes fixed on its next victim, you seize the opportunity to strike. With your polearm at the ready, you take advantage of the beast's momentary distraction to launch a swift and deadly attack. The air is electric with tension as the patrons of the Yawning Portal Tavern watch in horror, frozen in place by the unfolding drama.  You roll a d20 for your Attack Roll, adding your attack bonus with your polearm to the result. The dice clatters across the wooden table, coming to rest with...
- **Worst example**: ROUGE-L=0.0229 | LLM Composite=4.25
  - Input: !cast mass -t kei -t Ed -t Ceph -t Lav
  - Output: As the chaos of the Yawning Portal Tavern reaches a fever pitch, with the basilisk's deadly gaze sweeping across the room and the sounds of clashing steel filling the air, you focus your mind and attempt to cast a spell. The words "!cast mass -t kei -t Ed -t Ceph -t Lav" escape your lips, and a faint, magical glow begins to emanate from your outstretched hand.  The air around you seems to vibrate with anticipation as you channel the raw magical energy. However, the Rules Arbiter's voice echoes in your mind, remindi...

### No RAG

- **Best example**: ROUGE-L=0.105 | LLM Composite=4.75
  - Input: ) *The basilisk moves forward, now only with 6 targets to chose from* *Attack of Opportunity from Polearm Master as it moves forward.*
  - Output: As the basilisk moves forward, its unblinking eyes scan the room, searching for its next victim. The air is thick with tension, and the patrons of the Yawning Portal Tavern hold their collective breath. The bard in the corner raises an eyebrow, intrigued by the sudden turn of events. Durnan, the barkeep, polishes a mug with a dirty apron, his eyes fixed on the basilisk with a mixture of curiosity and concern.  With only six targets to choose from, the basilisk's gaze settles on the nearest adventurer, its eyes lock...
- **Worst example**: ROUGE-L=0.0 | LLM Composite=4.5
  - Input: *seeing as I am still stealthed, draws out shortbow*
  - Output: As you draw out your shortbow, the soft glow of the tavern's lanterns casts a warm light on the intricately carved wooden limbs, the string humming softly as you prepare to take aim. The air is thick with the smell of ale and roasting meat, the sounds of hushed conversations and clinking glasses filling the background. Durnan, the barkeep, polishes a mug with a white apron, his eyes scanning the room with a mixture of curiosity and wariness.  You remain stealthed, a ghostly figure blending into the shadows, unnotic...

## 3) Reflective analysis (what worked / what did not)

We compare each ablation to the baseline using changes in ROUGE-L (surface overlap), Rule Coverage (mechanics presence), and the rubric-based LLM composite.

- **Bare DM Only vs baseline**: ΔROUGE-L=+0.006, ΔRuleCoverage=-0.207, ΔLLMComposite=-0.375 → mechanics grounding appears to drop; overall rubric score worsens.
- **Local FT (No RAG) vs baseline**: ΔROUGE-L=+0.024, ΔRuleCoverage=-0.250, ΔLLMComposite=-2.719 → mechanics grounding appears to drop; overall rubric score worsens.
- **Local FT (RAG On) vs baseline**: ΔROUGE-L=+0.026, ΔRuleCoverage=-0.250, ΔLLMComposite=-3.219 → mechanics grounding appears to drop; overall rubric score worsens.
- **No Memory vs baseline**: ΔROUGE-L=+0.009, ΔRuleCoverage=-0.011, ΔLLMComposite=-0.031 → performance is relatively stable on the main signals.
- **No NPC Pass vs baseline**: ΔROUGE-L=+0.003, ΔRuleCoverage=-0.033, ΔLLMComposite=-0.062 → performance is relatively stable on the main signals.
- **No RAG vs baseline**: ΔROUGE-L=+0.004, ΔRuleCoverage=-0.190, ΔLLMComposite=-0.156 → mechanics grounding appears to drop.

### Concrete improvement direction for future work

Based on the observed deltas, a practical next step is to reduce the dependency on a single retrieval signal by improving SRD grounding (e.g., caching retrieved rules per query, using higher `k` plus reranking, and adding guardrails so the DM explicitly references the retrieved mechanics). In parallel, add a deterministic evaluation resume/caching strategy to further reduce Groq cost/rate-limit impact during repeated ablation runs.

## 4) Risks and mitigations

- **API rate limits / token budget exhaustion**: mitigate by running experiments sequentially, resuming from partial JSON outputs, and preferring local configurations when possible.
- **Telemetry / dependency incompatibilities** (e.g., Chroma telemetry): mitigate by disabling telemetry (see README) and pinning compatible NumPy versions.
- **Evaluation reliability**: mitigate by using multiple complementary metrics (surface overlap + semantic similarity + mechanics presence + rubric-based LLM judge) and sampling outputs qualitatively.
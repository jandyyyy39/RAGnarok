# Ablation Study Report (eval_crd3_ablation)

**Baseline:** Full Pipeline (Groq)

## 1) Quantitative evaluation

Table summarizes mean metrics across the evaluated inputs (typically 8 for the final run).

| Experiment | ROUGE-L | BERTScore | Rule Coverage | chrF | Latency (ms) | LLM Composite | Safety Pass Rate | Expected Groq Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bare DM Only | 0.054 | 0.811 | 0.049 | 0.119 | 2364 | 4.406 | 1.000 | 2 |
| Full Pipeline (Groq) | 0.057 | 0.809 | 0.234 | 0.125 | 1990 | 4.750 | 1.000 | 4 |
| Local FT (No RAG) | 0.138 | 0.838 | 0.005 | 0.207 | 8769 | 3.531 | 1.000 | 2 |
| Local FT (RAG On) | 0.113 | 0.836 | 0.016 | 0.220 | 11412 | 3.281 | 1.000 | 3 |
| No Memory | 0.056 | 0.810 | 0.245 | 0.120 | 2605 | 4.688 | 1.000 | 4 |
| No NPC Pass | 0.060 | 0.810 | 0.261 | 0.128 | 5207 | 4.750 | 1.000 | 3 |
| No RAG | 0.052 | 0.805 | 0.082 | 0.113 | 1873 | 4.312 | 1.000 | 3 |

### Cost / latency proxy

True monetary cost requires provider-specific token pricing and token counts. For reproducibility, we use latency plus an *expected Groq call count* derived from which agents are enabled (Rules Arbiter, DM Agent, NPC Consistency, and LLM-as-judge). This gives a stable, comparative proxy for cost.

## 2) Qualitative analysis (example runs)

### Bare DM Only

- **Best example**: ROUGE-L=0.0619 | LLM Composite=4.75
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: As you throw the punch, the sound of crunching metal and snapping leather fills the air, like the growl of a beast unleashed. Your fist, clad in the motorcycle helmet, connects with a sickening thud, sending shockwaves through the helmet and into your arm. The impact is akin to being on the receiving end of a brutal tackle, the force of which threatens to rattle your very bones.  The helmet, now a makeshift gauntlet, has amplified the impact of your blow, and your opponent staggers backward, eyes wide with surprise...
- **Worst example**: ROUGE-L=0.0455 | LLM Composite=3.0
  - Input: Marisha Ray! [cheering] | Mr. Taliesin Jaffe! [cheering] | Love you. Ashly Burch! [cheering] | Hide your kids, hide your wives, hide your faces. It's Sam Riegel. [cheering] | And-- Oh no. And your, my, our beloved Dungeo...
  - Output: As the player's voice booms through the land, the scene erupts into a frenzy of cheers and applause. The crowd, a mix of patrons and adventurers, joins in, whistling and stomping their feet. The air is electric, filled with an infectious energy that seems to reverberate through every molecule. The very walls of the establishment appear to vibrate with the collective enthusiasm, as if the sheer force of the player's admiration might shake the rafters loose.  Marisha Ray, Taliesin Jaffe, Ashly Burch, Sam Riegel, and ...

### Full Pipeline (Groq)

- **Best example**: ROUGE-L=0.0748 | LLM Composite=4.75
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: As you utter those words, "It's like getting punched with a motorcycle helmet on. Seven damage," the atmosphere in the Yawning Portal Tavern doesn't quite shift in response. The patrons continue their lively conversations, the clinking of glasses and the murmur of hushed talks filling the air. Durnan the Barkeep, polishing a mug with a white apron, raises an eyebrow but doesn't interrupt his task. It seems your statement, devoid of context within the bustling tavern, falls flat.  Mechanically, the Rules Arbiter has...
- **Worst example**: ROUGE-L=0.0748 | LLM Composite=4.75
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: As you utter those words, "It's like getting punched with a motorcycle helmet on. Seven damage," the atmosphere in the Yawning Portal Tavern doesn't quite shift in response. The patrons continue their lively conversations, the clinking of glasses and the murmur of hushed talks filling the air. Durnan the Barkeep, polishing a mug with a white apron, raises an eyebrow but doesn't interrupt his task. It seems your statement, devoid of context within the bustling tavern, falls flat.  Mechanically, the Rules Arbiter has...

### Local FT (No RAG)

- **Best example**: ROUGE-L=0.1515 | LLM Composite=4.25
  - Input: Oh yeah, that hits. For sure.
  - Output: The creature bursts out into this bright, white light, and it looks like there is a faint shimmer to its form as it just emerges and begins to drift in the air about a foot off of the ground.
- **Worst example**: ROUGE-L=0.209 | LLM Composite=2.0
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: It bursts out into this bright, white light, and it looks like there is a faint shimmer to its form as it drifts about a foot off of the ground in the center of this chamber.

### Local FT (RAG On)

- **Best example**: ROUGE-L=0.0893 | LLM Composite=4.25
  - Input: Oh yeah, that hits. For sure.
  - Output: All right, so. The creature bursts out into this bright light. You guys can't help but step back a little bit as you watch it emerge in this burst of radiant energy. As it does, the rest of you catch up with your eyes adjusting to the new surroundings. You can see before you this kind of glowing crystal-like structure that is just barely visible through the darkness of the cavern. The creature's head turns towards you and its jaws open wide.
- **Worst example**: ROUGE-L=0.1 | LLM Composite=2.0
  - Input: Marisha Ray! [cheering] | Mr. Taliesin Jaffe! [cheering] | Love you. Ashly Burch! [cheering] | Hide your kids, hide your wives, hide your faces. It's Sam Riegel. [cheering] | And-- Oh no. And your, my, our beloved Dungeo...
  - Output: As you guys begin to look over, it does not appear that there is any physical wound or damage to this creature from what you can see.

### No Memory

- **Best example**: ROUGE-L=0.0743 | LLM Composite=4.75
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: As you charge forward, the dimly lit chamber echoes with the sound of clashing steel and the stench of sweat and smoke. Your opponent, a hulking mass of muscle and fury, stands before you, its eyes blazing with a feral intensity. The air is heavy with tension, the weight of the battle hanging precariously in the balance.  With a fierce battle cry, you unleash a powerful blow, your fist flying toward the creature's jaw like a piston-driven hammer. The impact is tremendous, the sound of flesh meeting flesh a sickenin...
- **Worst example**: ROUGE-L=0.037 | LLM Composite=4.25
  - Input: Marisha Ray! [cheering] | Mr. Taliesin Jaffe! [cheering] | Love you. Ashly Burch! [cheering] | Hide your kids, hide your wives, hide your faces. It's Sam Riegel. [cheering] | And-- Oh no. And your, my, our beloved Dungeo...
  - Output: As you stand atop the worn, wooden stage in the midst of the bustling tavern, the crowd's attention is drawn to you like a magnet. The air is thick with the smell of ale and roasting meat, and the fire crackling in the hearth casts a warm, golden glow over the sea of faces. With a flair for the dramatic, you launch into a passionate tribute, extolling the virtues of the esteemed personalities you've named: Marisha Ray, Taliesin Jaffe, Ashly Burch, Sam Riegel, and the incomparable Matthew Mercer.  Your words are met...

### No NPC Pass

- **Best example**: ROUGE-L=0.0748 | LLM Composite=4.75
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: As you stand amidst the murmuring patrons of the Yawning Portal Tavern, the air thick with anticipation, you focus your energy on the task at hand. With a deep breath, you raise your Magician's Judge, and the room seems to hold its collective breath in expectation. The patrons' chatter dies down, and all eyes are on you as you prepare to unleash the Dispel Magic portion of your Judge.  With a swift motion, you strike the target, and the sound of the impact is akin to getting punched with a motorcycle helmet on – a ...
- **Worst example**: ROUGE-L=0.0748 | LLM Composite=4.75
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: As you stand amidst the murmuring patrons of the Yawning Portal Tavern, the air thick with anticipation, you focus your energy on the task at hand. With a deep breath, you raise your Magician's Judge, and the room seems to hold its collective breath in expectation. The patrons' chatter dies down, and all eyes are on you as you prepare to unleash the Dispel Magic portion of your Judge.  With a swift motion, you strike the target, and the sound of the impact is akin to getting punched with a motorcycle helmet on – a ...

### No RAG

- **Best example**: ROUGE-L=0.0302 | LLM Composite=4.75
  - Input: No more of you doing my voice, Matt?
  - Output: As you utter the words, "No more of you doing my voice, Matt?" the atmosphere in the Yawning Portal Tavern takes a subtle yet palpable shift. The air seems to vibrate with a mix of amusement and anticipation, as if the patrons are collectively holding their breath, waiting to see how the situation will unfold. Durnan, the barkeep, looks up from polishing a mug, his eyes twinkling with interest, his bushy eyebrows raised in a silent query.  The patrons, still reeling from the earlier commotion, exchange furtive glan...
- **Worst example**: ROUGE-L=0.0557 | LLM Composite=3.25
  - Input: Okay. | It's like getting punched with a motorcycle helmet on. | Seven damage.
  - Output: As you utter the phrase, "Okay. It's like getting punched with a motorcycle helmet on. Seven damage," the atmosphere within the Yawning Portal Tavern takes a peculiar turn. The air, already thick with the tension of your intensified stink-eye, seems to ripple with anticipation. The patrons, still reeling from the earlier outburst of cheering and applause for the esteemed Dungeon Master, Matthew Mercer, now lean in with a mix of curiosity and apprehension.  Durnan the Barkeep, ever vigilant, watches the scene unfold...

## 3) Reflective analysis (what worked / what did not)

We compare each ablation to the baseline using changes in ROUGE-L (surface overlap), Rule Coverage (mechanics presence), and the rubric-based LLM composite.

- **Bare DM Only vs baseline**: ΔROUGE-L=-0.002, ΔRuleCoverage=-0.185, ΔLLMComposite=-0.344 → mechanics grounding appears to drop; overall rubric score worsens.
- **Local FT (No RAG) vs baseline**: ΔROUGE-L=+0.081, ΔRuleCoverage=-0.228, ΔLLMComposite=-1.219 → mechanics grounding appears to drop; overall rubric score worsens.
- **Local FT (RAG On) vs baseline**: ΔROUGE-L=+0.056, ΔRuleCoverage=-0.217, ΔLLMComposite=-1.469 → mechanics grounding appears to drop; overall rubric score worsens.
- **No Memory vs baseline**: ΔROUGE-L=-0.000, ΔRuleCoverage=+0.011, ΔLLMComposite=-0.062 → performance is relatively stable on the main signals.
- **No NPC Pass vs baseline**: ΔROUGE-L=+0.004, ΔRuleCoverage=+0.027, ΔLLMComposite=+0.000 → performance is relatively stable on the main signals.
- **No RAG vs baseline**: ΔROUGE-L=-0.005, ΔRuleCoverage=-0.152, ΔLLMComposite=-0.438 → mechanics grounding appears to drop; overall rubric score worsens.

### Concrete improvement direction for future work

Based on the observed deltas, a practical next step is to reduce the dependency on a single retrieval signal by improving SRD grounding (e.g., caching retrieved rules per query, using higher `k` plus reranking, and adding guardrails so the DM explicitly references the retrieved mechanics). In parallel, add a deterministic evaluation resume/caching strategy to further reduce Groq cost/rate-limit impact during repeated ablation runs.

## 4) Risks and mitigations

- **API rate limits / token budget exhaustion**: mitigate by running experiments sequentially, resuming from partial JSON outputs, and preferring local configurations when possible.
- **Telemetry / dependency incompatibilities** (e.g., Chroma telemetry): mitigate by disabling telemetry (see README) and pinning compatible NumPy versions.
- **Evaluation reliability**: mitigate by using multiple complementary metrics (surface overlap + semantic similarity + mechanics presence + rubric-based LLM judge) and sampling outputs qualitatively.
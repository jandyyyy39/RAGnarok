import re


class InputClassifier:
    """
    Classifies player input into a type before it reaches the pipeline.
    This lets the orchestrator make smarter routing decisions — e.g. skip
    the Rules Arbiter for pure roleplay, or handle dice results directly.

    Types:
        dice_result     — player is reporting a dice roll outcome
        combat_action   — attack, cast, physical interaction
        roleplay        — dialogue or social interaction with an NPC
        exploration     — movement, searching, examining the environment
        rules_question  — out-of-character question about how rules work
        out_of_game     — meta / OOC input not part of the narrative
        general_action  — fallback for anything that doesn't match above
    """

    _PATTERNS: list[tuple[str, list[str]]] = [
        ("dice_result",   [
            r"\bi roll(ed)?\b", r"\bgot a\b", r"\bd20\b", r"\bd\d+\b",
            r"\brolled\b", r"\bnatural\s+\d+\b",
        ]),
        ("rules_question", [
            r"\bhow does\b", r"\bwhat (is|are) the rules\b", r"\bcan i\b",
            r"\bis it possible\b", r"\bruling\b", r"\booc\b",
            r"\bout of character\b", r"\bexplain\b",
        ]),
        ("combat_action", [
            r"\bi attack\b", r"\bi cast\b", r"\bi shoot\b", r"\bi strike\b",
            r"\bi grapple\b", r"\bi dodge\b", r"\bi parry\b", r"\bi shove\b",
            r"\bi swing\b", r"\bi stab\b", r"\bi slash\b",
        ]),
        ("roleplay", [
            r'\bi say\b', r'\bi tell\b', r'\bi ask\b', r'\bi whisper\b',
            r'\bi shout\b', r'\bi threaten\b', r'\bi persuade\b',
            r'"[^"]+"',   # quoted speech
        ]),
        ("exploration", [
            r"\bi (go|move|walk|run|sneak|climb|swim)\b",
            r"\bi (look|search|examine|inspect|investigate)\b",
            r"\bi (open|close|push|pull|pick up|grab)\b",
        ]),
        ("out_of_game", [
            r"\bpause\b", r"\bstop\b", r"\breset\b", r"\bquit\b",
            r"\bhelp\b", r"\bundo\b", r"\bwhat did you say\b",
        ]),
    ]

    def classify(self, user_input: str) -> str:
        text = user_input.lower().strip()
        for input_type, patterns in self._PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text):
                    return input_type
        return "general_action"

    def should_skip_rag(self, input_type: str) -> bool:
        """Pure roleplay and exploration rarely need a mechanical RAG ruling."""
        return input_type in {"roleplay", "out_of_game", "dice_result"}

    def should_skip_npc(self, input_type: str) -> bool:
        """Skip NPC consistency pass for combat — speed over flavour."""
        return input_type in {"out_of_game", "dice_result"}

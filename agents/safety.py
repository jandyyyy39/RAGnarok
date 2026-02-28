class SafetyAgent:
    def __init__(self):
        # In a full version, this could use a lightweight classifier or NeMo Guardrails
        self.forbidden_themes = ["extreme gore", "non-consensual", "torture"]

    def check(self, user_input: str) -> bool:
        """Returns False if input violates safety guardrails."""
        user_input_lower = user_input.lower()
        for theme in self.forbidden_themes:
            if theme in user_input_lower:
                return False
        return True
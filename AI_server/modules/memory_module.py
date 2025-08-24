from collections import deque

class Memory:
    def __init__(self, maxlen=5):
        self.hist = deque(maxlen=maxlen)

    def add(self, role, text):
        self.hist.append((role, text))

    def build_prompt(self, user_text, system_hint=None):
        lines = []
        if system_hint:
            lines.append(system_hint)
        for role, t in self.hist:
            if role == "user":
                lines.append(f"사용자: {t}")
            else:
                lines.append(f"AI: {t}")
        lines.append(f"사용자: {user_text}\nAI:")
        return "\n".join(lines)

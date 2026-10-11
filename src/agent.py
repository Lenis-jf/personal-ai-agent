from llm.base import LLM

class Agent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def ask(self, user_message: str) -> str:
        prompt = ("You are a helpful personal computer assistant. "
            "Answer clearly and concisely.\n\n"
            f"User request: {user_message}\n")

        return self.llm.generate(prompt)
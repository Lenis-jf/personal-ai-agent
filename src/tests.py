from llm.local_llm_mlx import LocalLLMMLX
from agent import Agent

llm = LocalLLMMLX(
    "mlx-community/gemma-4-e4b-it-4bit"
)

agent = Agent(llm)

response = agent.ask(
    "Explain what an AI agent is in one short paragraph."
)

print(response)
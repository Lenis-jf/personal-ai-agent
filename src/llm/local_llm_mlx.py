from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

from .base import LLM

class LocalLLMMLX(LLM):

    def __init__(self, model_path: str):
        self.model, self.processor = load(model_path)

    def generate(self, prompt: str) -> str:
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
        )

        result = generate(
            model=self.model,
            processor=self.processor,
            prompt=formatted_prompt,
            max_tokens=256,
            temperature=0.0,
        )

        return result.text
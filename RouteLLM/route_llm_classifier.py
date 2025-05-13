import os
import re

os.environ["HF_TOKEN"] = "empty"
os.environ["OPENAI_API_KEY"] = "empty"
os.environ["VLLM_API_KEY"] = "empty"
os.environ["GEMINI_API_KEY"] = "empty"
os.environ["TOGETHERAI_API_KEY"] = "empty"

os.environ["TOKENIZERS_PARALLELISM"] = "false"     

from routellm.controller import Controller

class RouteLLMClassifier:
    def __init__(self, strong_model, weak_model):
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.client = Controller(
            routers=["mf"],
            strong_model = self.strong_model,
            weak_model = self.weak_model,
        )

    def classify(self, text):
        response = self.client.chat.completions.create(
            # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
            model="router-mf-0.11593",
            messages=[
                {"role": "user", "content": text}
            ]
        )
        return response

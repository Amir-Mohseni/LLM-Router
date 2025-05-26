import os
import re
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")

# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
# print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
# print("TOGETHERAI_API_KEY:", os.getenv("TOGETHERAI_API_KEY"))

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

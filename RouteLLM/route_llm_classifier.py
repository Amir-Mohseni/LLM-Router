from dotenv import load_dotenv
load_dotenv()

from routellm.controller import Controller

class RouteLLMClassifier:
    def __init__(self, strong_model, weak_model, threshold=0.11593):
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.threshold = threshold
        self.client = Controller(
            routers=["mf"],
            strong_model=self.strong_model,
            weak_model=self.weak_model,
        )

    def classify(self, text):
        try:
            model_name = f"router-mf-{self.threshold:.5f}"
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": text}]
            )
            return response.choices[0].message["content"]
        except Exception as e:
            print(f"Classification error: {e}")
            return None

from dotenv import load_dotenv
load_dotenv()

from routellm.controller import Controller
import os

class RouteLLMClassifier:
    def __init__(self, strong_model, weak_model, api_base=None, api_key=None, threshold=0.11593, router_type="bert"):
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.threshold = threshold
        self.router_type = router_type
        
        self.client = Controller(
            routers=[router_type],
            strong_model=self.strong_model,
            weak_model=self.weak_model,
        )
        if api_base is not None:
            self.client.api_base = api_base
        if api_key is not None:
            self.client.api_key = api_key

    def predict_class(self, text):
        """
        Predict which model class (strong/weak) to route to for the given text.
        Returns: 'strong' or 'weak'
        """
        try:
            routed_model = self.client.route(
                prompt=text,
                router=self.router_type, 
                threshold=self.threshold
            )
            
            # Return simplified class labels
            if routed_model == self.strong_model:
                return "strong"
            else:
                return "weak"
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_confidence_score(self, text):
        """
        Get the confidence score (0-1) for routing to the strong model.
        Higher values indicate higher confidence that strong model should be used.
        """
        try:
            confidence = self.client.routers[self.router_type].calculate_strong_win_rate(text)
            return confidence
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return None

    def classify(self, text):
        try:
            model_name = f"router-{self.router_type}-{self.threshold:.5f}"
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": text}]
            )
            return response.choices[0].message["content"]
        except Exception as e:
            print(f"Classification error: {e}")
            return None

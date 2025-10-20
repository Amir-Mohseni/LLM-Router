"""
RouteLLM router implementation for evaluation.
Uses the routellm library directly without custom wrappers.
"""

from routellm.controller import Controller
from typing import Optional


class RouteLLMRouter:
    """Wrapper for RouteLLM confidence-based routing."""
    
    def __init__(
        self, 
        strong_model: str, 
        weak_model: str, 
        router_type: str = "bert"
    ):
        """
        Initialize RouteLLM router.
        
        Args:
            strong_model: Identifier for the strong model
            weak_model: Identifier for the weak model
            router_type: Type of router ("bert", "mf", "sw_ranking", etc.)
        """
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.router_type = router_type
        
        # Initialize controller with the specified router
        self.controller = Controller(
            routers=[router_type],
            strong_model=strong_model,
            weak_model=weak_model,
        )
    
    def get_confidence_score(self, prompt: str) -> float:
        """
        Get confidence score for routing to strong model.
        
        Args:
            prompt: Input text to evaluate
            
        Returns:
            Float between 0 and 1, where higher values indicate
            higher confidence that the strong model should be used.
        """
        try:
            confidence = self.controller.routers[self.router_type].calculate_strong_win_rate(prompt)
            return confidence
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5  # Default to neutral
    
    def route(self, prompt: str, threshold: float = 0.5) -> str:
        """
        Route prompt to strong or weak model based on threshold.
        
        Args:
            prompt: Input text to route
            threshold: Confidence threshold for routing to strong model
            
        Returns:
            "strong" or "weak"
        """
        confidence = self.get_confidence_score(prompt)
        return "strong" if confidence >= threshold else "weak"


import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Tuple
import os
from huggingface_hub import hf_hub_download
# Remove HfHubDownloadError if it's not available in your version
# from huggingface_hub import hf_hub_download, HfHubDownloadError # <--- REMOVE HfHubDownloadError
from safetensors.torch import load_file # Import load_file for .safetensors
import requests # Import requests to catch its exceptions

class RouterMultiTaskModel(nn.Module):
    def __init__(self, base_model_id: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_id)
        hidden_size = self.base.config.hidden_size

        for param in self.base.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.reasoning_head = nn.Linear(hidden_size, 1)
        self.difficulty_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None) -> Dict[str, torch.Tensor]:
        base_output = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = base_output.last_hidden_state[:, 0]
        pooled = self.dropout(cls_token)

        return {
            "reasoning_score": torch.sigmoid(self.reasoning_head(pooled)).squeeze(-1),
            "difficulty_score": torch.sigmoid(self.difficulty_head(pooled)).squeeze(-1)
        }


class MultiTaskRouter:
    """
    Router that uses a multi-task transformer model to determine routing decisions.
    Provides two scores: reasoning_score and difficulty_score to guide model selection.
    """
    
    def __init__(self, base_model_name: str = 'Qwen/Qwen3-0.6B', router_model_name: str = 'AmirMohseni/LLM-Router-v1'):
        """
        Initialize the multi-task router
        
        Args:
            base_model_name: The base transformer model to use (default: 'Qwen/Qwen3-0.6B')
            router_model_name: Path to the trained router model weights (default: 'AmirMohseni/LLM-Router-v1')
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = RouterMultiTaskModel(base_model_name)

        print(f"Attempting to load router weights from {router_model_name}...")
        try:
            router_state_dict = load_file(
                hf_hub_download(router_model_name, filename="model.safetensors"),
                device=self.device
            )

            head_mappings = {
                "reasoning_head": self.model.reasoning_head,
                "difficulty_head": self.model.difficulty_head,
            }

            loaded_any_weights = False
            for head_name, head_module in head_mappings.items():
                weight_key = f"{head_name}.weight"
                bias_key = f"{head_name}.bias"

                if weight_key in router_state_dict and bias_key in router_state_dict:
                    head_state = {
                        "weight": router_state_dict[weight_key],
                        "bias": router_state_dict[bias_key]
                    }
                    head_module.load_state_dict(head_state)
                    print(f"Successfully loaded '{head_name}' weights into RouterMultiTaskModel.")
                    loaded_any_weights = True
                else:
                    print(f"Warning: '{head_name}' weights (or bias) not found in router model state_dict. "
                          f"This head will remain randomly initialized.")
            
            if not loaded_any_weights:
                print(f"No specific head weights were loaded from {router_model_name}. "
                      "Ensure key names match or the router model contains the expected heads.")

        # --- MODIFIED EXCEPTION HANDLING ---
        except requests.exceptions.RequestException as e: # Catch network/download errors
            print(f"Error downloading router model (network/HTTP issue): {e}")
            print("Router weights could not be loaded. Reasoning and difficulty heads will use randomly initialized weights.")
        except FileNotFoundError as e: # Catch if hf_hub_download returns a path that doesn't exist
            print(f"Error: Safetensors file not found after download attempt: {e}")
            print("Router weights could not be loaded. Reasoning and difficulty heads will use randomly initialized weights.")
        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred while loading router weights: {e}")
            print("Router weights could not be loaded. Reasoning and difficulty heads will use randomly initialized weights.")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.model_mapping = {
            (True, True): "google/gemini-2.5-pro-preview",
            (True, False): "qwen/qwen3-14b",
            (False, True): "google/gemini-2.0-flash-001",
            (False, False): "google/gemma-3-4b-it"
        }
    
    def route(self, text: str, reasoning_threshold: float = 0.4, 
              difficulty_threshold: float = 0.5) -> Dict[str, Any]:
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=2048
            ).to(self.device)
            
            outputs = self.model(**inputs)
            reasoning_score = outputs["reasoning_score"].item()
            difficulty_score = outputs["difficulty_score"].item()
            
            print(f"Reasoning score: {reasoning_score}, Difficulty score: {difficulty_score}")
            
            reasoning_needed = reasoning_score >= reasoning_threshold
            high_difficulty = difficulty_score >= difficulty_threshold
            
            selected_model = self.model_mapping[(reasoning_needed, high_difficulty)]
            
            return {
                "selected_model": selected_model,
                "reasoning_score": reasoning_score,
                "difficulty_score": difficulty_score,
                "reasoning_needed": reasoning_needed,
                "high_difficulty": high_difficulty,
            }
    
    def get_available_models(self) -> list:
        return list(set(self.model_mapping.values()))
    
    def get_model_scores(self, text: str) -> Tuple[float, float]:
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2048
            ).to(self.device)
            
            outputs = self.model(**inputs)
            return outputs["reasoning_score"].item(), outputs["difficulty_score"].item() 
        
if __name__ == "__main__":
    print("Initializing MultiTaskRouter...")
    router = MultiTaskRouter()
    print("\nRouter initialized. Performing routing example:")

    test_queries = [
        "What is the capital of France?",
        "Explain the theory of general relativity in simple terms.",
        "Implement a quicksort algorithm in Python and analyze its time complexity.",
        "Name 5 colors."
    ]

    for query in test_queries:
        print(f"\nRouting query: \"{query}\"")
        route_decision = router.route(query)
        print(f"  Selected Model: {route_decision['selected_model']}")
        print(f"  Reasoning Score: {route_decision['reasoning_score']:.4f} (Needed: {route_decision['reasoning_needed']})")
        print(f"  Difficulty Score: {route_decision['difficulty_score']:.4f} (High: {route_decision['high_difficulty']})")
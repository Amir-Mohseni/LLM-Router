"""
Efficient router evaluation script with configurable settings.
Batch processing and JSONL output.
"""

import os
import json
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Dict, List, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # If it's a relative path, make it relative to the script's directory
    if not os.path.isabs(config_path):
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_device(preference: str = "auto") -> torch.device:
    """Detect and return the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(preference)


class RouterEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = detect_device(config["device"]["preference"])
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        model_id = config["model"]["id"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        # Labels mapping
        self.labels = config["labels"]
        
    def predict_batch(self, prompts: List[str]) -> List[Dict[str, float]]:
        """Predict probabilities for a batch of prompts."""
        # Tokenize batch
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.config["processing"]["max_length"]
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Extract scores for each sample
        results = []
        for i in range(len(prompts)):
            scores = {
                f"{self.labels[j]}_score": float(probs[i, j]) 
                for j in range(len(self.labels))
            }
            results.append(scores)
        
        return results
    
    def evaluate_dataset(self) -> None:
        """Evaluate the entire dataset and save results."""
        # Load dataset
        dataset_config = self.config["dataset"]
        dataset = load_dataset(dataset_config["name"], split=dataset_config["split"])
        
        # Create organized output directory structure
        # Extract model name (remove username/org prefix)
        model_name = self.config["model"]["id"].split("/")[-1]
        
        # Extract dataset name (remove username/org prefix)
        dataset_name = dataset_config["name"].split("/")[-1]
        
        # Create nested directory structure: results/dataset_name/model_name/
        output_dir = Path(self.config["output"]["results_dir"]) / dataset_name / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / self.config["output"]["filename"]
        
        # Process in batches
        batch_size = self.config["processing"]["batch_size"]
        prompt_column = dataset_config["prompt_column"]
        
        results = []
        
        print(f"Processing {len(dataset)} samples in batches of {batch_size}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Output directory: {output_dir}")
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch_end = min(i + batch_size, len(dataset))
            batch_data = dataset[i:batch_end]
            
            # Extract prompts
            if isinstance(batch_data[prompt_column], list):
                prompts = batch_data[prompt_column]
            else:
                prompts = [batch_data[prompt_column]]
            
            # Get predictions
            predictions = self.predict_batch(prompts)
            
            # Combine with original data
            for j, prediction in enumerate(predictions):
                idx = i + j
                result = dict(dataset[idx])  # Copy original data
                result.update(prediction)  # Add prediction scores
                results.append(result)
        
        # Save to JSONL
        print(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Evaluation complete! Results saved to {output_file}")
        
        # Print summary statistics
        think_scores = [r['think_score'] for r in results]
        no_think_scores = [r['no_think_score'] for r in results]
        
        print(f"\nSummary:")
        print(f"Total samples: {len(results)}")
        print(f"Average think_score: {sum(think_scores) / len(think_scores):.4f}")
        print(f"Average no_think_score: {sum(no_think_scores) / len(no_think_scores):.4f}")
        print(f"Samples classified as 'think': {sum(1 for s in think_scores if s > 0.5)}")
        print(f"Samples classified as 'no_think': {sum(1 for s in no_think_scores if s > 0.5)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate router on dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize evaluator
    evaluator = RouterEvaluator(config)
    
    # Run evaluation
    evaluator.evaluate_dataset()


if __name__ == "__main__":
    main()

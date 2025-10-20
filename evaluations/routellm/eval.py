"""
Evaluator for RouteLLM confidence-based routers.
Performs threshold analysis to compare routing strategies.
Supports evaluation with model inference results (accuracy/reward-based).
"""

import os
import json
import yaml
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from dotenv import load_dotenv
from router import RouteLLMRouter

load_dotenv()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class RouteLLMEvaluator:
    """Evaluator for confidence-based routers with threshold analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize router
        router_config = config["router"]
        self.router = RouteLLMRouter(
            strong_model=router_config["strong_model"],
            weak_model=router_config["weak_model"],
            router_type=router_config.get("type", "bert")
        )
        
    def evaluate_dataset(self) -> None:
        """Evaluate dataset and perform threshold analysis."""
        # Load dataset
        dataset_config = self.config["dataset"]
        dataset = load_dataset(dataset_config["name"], split=dataset_config["split"])
        
        # Create output directory
        router_name = f"routellm-{self.config['router']['type']}"
        dataset_name = dataset_config["name"].split("/")[-1]
        output_dir = Path(self.config["output"]["results_dir"]) / dataset_name / router_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / self.config["output"]["filename"]
        
        # Process samples
        prompt_column = dataset_config["prompt_column"]
        results = []
        
        print(f"\n{'='*80}")
        print(f"RouteLLM Evaluation")
        print(f"{'='*80}")
        print(f"Processing {len(dataset)} samples")
        print(f"Router: {router_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")
        
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[i]
            prompt = sample[prompt_column]
            
            # Get confidence score
            confidence = self.router.get_confidence_score(prompt)
            
            # Convert to Python float to avoid JSON serialization issues
            confidence = float(confidence)
            
            # Combine with original data
            # Use probability names to avoid overwriting dataset's think_score/no_think_score
            result = dict(sample)
            result.update({
                "no_think_probability": 1.0 - confidence,  # Lower confidence = simpler = no_think
                "think_probability": confidence             # Higher confidence = complex = think
            })
            results.append(result)
        
        # Save raw results
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Perform threshold analysis
        self._threshold_analysis(results, dataset_config, output_dir)
        
        # Print summary
        self._print_summary(results, dataset_config)
        
        print(f"Evaluation complete! Results saved to {output_file}\n")
    
    def _threshold_analysis(self, results: List[Dict], dataset_config: Dict, output_dir: Path):
        """Analyze routing behavior across different thresholds."""
        think_probabilities = [r['think_probability'] for r in results]
        
        # Check if we have labels (from diff_score analysis)
        has_labels = 'label' in results[0] if results else False
        labels = [r.get('label') for r in results] if has_labels else None
        
        # Test thresholds from 0.1 to 0.9
        thresholds = [i/100 for i in range(10, 91, 5)]
        threshold_results = []
        
        print(f"\n{'='*80}")
        if has_labels:
            print(f"Threshold Analysis (with Accuracy)")
        else:
            print(f"Threshold Analysis (Distribution Only)")
        print(f"{'='*80}")
        
        if has_labels:
            print(f"{'Threshold':<12} {'Accuracy':<12} {'Think %':<12} {'No-Think %':<12}")
        else:
            print(f"{'Threshold':<12} {'Think %':<12} {'No-Think %':<12}")
        print("-" * 80)
        
        best_threshold = None
        best_accuracy = 0.0
        
        for threshold in thresholds:
            # Predict 'think' if think_probability >= threshold, else 'no_think'
            predictions = ['think' if c >= threshold else 'no_think' for c in think_probabilities]
            
            think_pct = sum(1 for p in predictions if p == 'think') / len(predictions) * 100
            no_think_pct = 100 - think_pct
            
            result_entry = {
                "threshold": threshold,
                "think_percentage": think_pct,
                "no_think_percentage": no_think_pct,
                "think_count": int(think_pct * len(predictions) / 100),
                "no_think_count": int(no_think_pct * len(predictions) / 100)
            }
            
            # Calculate accuracy if labels are available
            if has_labels:
                correct = sum(1 for p, l in zip(predictions, labels) if p == l and l is not None)
                total = sum(1 for l in labels if l is not None)
                accuracy = correct / total if total > 0 else 0.0
                result_entry["accuracy"] = accuracy
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                
                print(f"{threshold:<12.2f} {accuracy:<12.4f} {think_pct:<12.1f}% {no_think_pct:<12.1f}%")
            else:
                print(f"{threshold:<12.2f} {think_pct:<12.1f}% {no_think_pct:<12.1f}%")
            
            threshold_results.append(result_entry)
        
        print("-" * 80)
        
        if has_labels and best_threshold is not None:
            print(f"\n✓ Best threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.4f})")
        
        # Save threshold analysis
        threshold_file = output_dir / "threshold_analysis.json"
        with open(threshold_file, 'w') as f:
            analysis_data = {
                "threshold_results": threshold_results
            }
            if has_labels:
                analysis_data["best_threshold"] = best_threshold
                analysis_data["best_accuracy"] = best_accuracy
            json.dump(analysis_data, f, indent=2)
        
        print(f"Threshold analysis saved to {threshold_file}")
        print(f"{'='*80}\n")
    
    def _print_summary(self, results: List[Dict], dataset_config: Dict) -> None:
        """Print summary statistics similar to router_eval.py."""
        think_probabilities = [r['think_probability'] for r in results]
        no_think_probabilities = [r['no_think_probability'] for r in results]
        
        print(f"\n{'='*80}")
        print(f"Summary Statistics")
        print(f"{'='*80}")
        print(f"Total samples: {len(results)}")
        print(f"Average think_probability: {sum(think_probabilities) / len(think_probabilities):.4f}")
        print(f"Average no_think_probability: {sum(no_think_probabilities) / len(no_think_probabilities):.4f}")
        print(f"Min think_probability: {min(think_probabilities):.4f}")
        print(f"Max think_probability: {max(think_probabilities):.4f}")
        
        # Classification at 0.5 threshold (default)
        think_count = sum(1 for s in think_probabilities if s > 0.5)
        no_think_count = sum(1 for s in no_think_probabilities if s > 0.5)
        print(f"\nClassification at 0.5 threshold:")
        print(f"  Samples classified as 'think': {think_count} ({think_count/len(results)*100:.1f}%)")
        print(f"  Samples classified as 'no_think': {no_think_count} ({no_think_count/len(results)*100:.1f}%)")
        
        # Check if results already have labels and diff_scores
        has_labels = 'label' in results[0] if results else False
        
        if has_labels:
            # Results already have diff_score analysis, show label-based summary
            self._print_label_summary(results)
        else:
            # Load model inference results for accuracy/reward analysis
            inference_results = self._load_inference_results(dataset_config)
            if inference_results is not None:
                self._print_inference_analysis(results, inference_results)
            else:
                print(f"\nNote: No inference results found. Add 'inference_results' to config for detailed analysis.")
        
        print(f"{'='*80}\n")
    
    def _load_inference_results(self, dataset_config: Dict) -> Optional[pd.DataFrame]:
        """Load model inference results if available."""
        if "inference_results" not in dataset_config:
            return None
        
        inference_config = dataset_config["inference_results"]
        dataset_name = dataset_config["name"].split("/")[-1]
        model_name = inference_config["model_name"]
        results_dir = Path(self.config["output"]["results_dir"])
        
        try:
            # Load no_think and think results
            no_think_path = results_dir / dataset_name / model_name / f"{model_name}-no-think.jsonl"
            think_path = results_dir / dataset_name / model_name / f"{model_name}-think.jsonl"
            
            if not (no_think_path.exists() and think_path.exists()):
                return None
            
            no_think_df = pd.read_json(no_think_path, lines=True)
            think_df = pd.read_json(think_path, lines=True)
            
            # Process responses for accuracy or use pre-computed scores
            if "responses" in no_think_df.columns:
                # Calculate accuracy from responses (for math datasets)
                no_think_df = self._process_responses(no_think_df, "no_think")
                think_df = self._process_responses(think_df, "think")
            
            # Merge on unique_id or index
            if "unique_id" in no_think_df.columns and "unique_id" in think_df.columns:
                merged = pd.merge(no_think_df, think_df, on="unique_id", suffixes=("", "_think"))
            else:
                merged = pd.concat([no_think_df.reset_index(drop=True), 
                                   think_df.reset_index(drop=True)], axis=1)
            
            # Clean up duplicate columns
            columns_to_drop = [col for col in merged.columns 
                             if col.endswith("_think") and col.replace("_think", "") in merged.columns]
            merged = merged.drop(columns=columns_to_drop)
            
            return merged
            
        except Exception as e:
            print(f"Warning: Could not load inference results: {e}")
            return None
    
    def _process_responses(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Extract accuracy statistics from responses column."""
        df = df.copy()
        df[f"{prefix}_correct_count"] = df["responses"].apply(
            lambda responses: sum(1 for response in responses if response.get("is_correct", False))
        )
        df[f"{prefix}_total_count"] = df["responses"].apply(len)
        df[f"{prefix}_accuracy"] = df[f"{prefix}_correct_count"] / df[f"{prefix}_total_count"]
        df = df.drop(columns=["responses"])
        return df
    
    def _print_inference_analysis(self, results: List[Dict], inference_df: pd.DataFrame) -> None:
        """Print analysis comparing router decisions with actual model performance."""
        print(f"\n{'='*80}")
        print(f"Router Performance Analysis (with model inference results)")
        print(f"{'='*80}")
        
        # Add router predictions to inference dataframe
        df = inference_df.copy()
        for idx, result in enumerate(results):
            if idx < len(df):
                df.loc[idx, 'router_think_probability'] = result['think_probability']
                df.loc[idx, 'router_no_think_probability'] = result['no_think_probability']
        
        df['router_pick'] = df.apply(
            lambda row: 'think' if row['router_think_probability'] >= row['router_no_think_probability'] else 'no_think',
            axis=1
        )
        
        # Check if we have accuracy metrics (math datasets) or score metrics (quality datasets)
        if 'no_think_accuracy' in df.columns and 'think_accuracy' in df.columns:
            # Accuracy-based evaluation (math datasets)
            no_think_avg = df['no_think_accuracy'].mean()
            think_avg = df['think_accuracy'].mean()
            
            df['router_accuracy'] = df.apply(
                lambda row: row['think_accuracy'] if row['router_pick'] == 'think' else row['no_think_accuracy'],
                axis=1
            )
            router_avg = df['router_accuracy'].mean()
            
            print(f"Average Accuracy:")
            print(f"  Always No-Think:  {no_think_avg:.4f}")
            print(f"  Always Think:     {think_avg:.4f}")
            print(f"  Router:           {router_avg:.4f}")
            print(f"\nRouter Improvement:")
            print(f"  vs Always No-Think:  {router_avg - no_think_avg:+.4f}")
            print(f"  vs Always Think:     {router_avg - think_avg:+.4f}")
            
        elif 'no_think_score' in df.columns and 'think_score' in df.columns:
            # Score/reward-based evaluation (quality datasets)
            no_think_avg = df['no_think_score'].mean()
            think_avg = df['think_score'].mean()
            
            df['router_reward'] = df.apply(
                lambda row: row['think_score'] if row['router_pick'] == 'think' else row['no_think_score'],
                axis=1
            )
            router_avg = df['router_reward'].mean()
            
            print(f"Average Reward/Score:")
            print(f"  Always No-Think:  {no_think_avg:.4f}")
            print(f"  Always Think:     {think_avg:.4f}")
            print(f"  Router:           {router_avg:.4f}")
            print(f"\nRouter Improvement:")
            print(f"  vs Always No-Think:  {router_avg - no_think_avg:+.4f}")
            print(f"  vs Always Think:     {router_avg - think_avg:+.4f}")
        
        print(f"\nRouter Pick Distribution:")
        print(df['router_pick'].value_counts().to_string())
    
    def _print_label_summary(self, results: List[Dict]) -> None:
        """Print summary when results already have labels from diff_score."""
        print(f"\n{'='*80}")
        print(f"Router Performance (based on diff_score labels)")
        print(f"{'='*80}")
        
        labels = [r.get('label') for r in results]
        router_think_probabilities = [r['think_probability'] for r in results]
        
        # Calculate accuracy at 0.5 threshold
        predictions = ['think' if s >= 0.5 else 'no_think' for s in router_think_probabilities]
        correct = sum(1 for p, l in zip(predictions, labels) if p == l and l is not None)
        total = sum(1 for l in labels if l is not None)
        accuracy_05 = correct / total if total > 0 else 0.0
        
        # Distribution of labels
        label_counts = {}
        for label in labels:
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Router picks at 0.5
        router_picks = {}
        for pred in predictions:
            router_picks[pred] = router_picks.get(pred, 0) + 1
        
        print(f"Label Distribution (ground truth from diff_score):")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({count/total*100:.1f}%)")
        
        print(f"\nRouter Picks at 0.5 threshold:")
        for pick, count in sorted(router_picks.items()):
            print(f"  {pick}: {count} ({count/len(predictions)*100:.1f}%)")
        
        print(f"\nAccuracy at 0.5 threshold: {accuracy_05:.4f} ({correct}/{total})")
        
        # Calculate average rewards using dataset's think_score and no_think_score
        if 'think_score' in results[0] and 'no_think_score' in results[0]:
            model_think_scores = [r.get('think_score') for r in results if r.get('think_score') is not None]
            model_no_think_scores = [r.get('no_think_score') for r in results if r.get('no_think_score') is not None]
            
            # Calculate average rewards for different strategies
            no_think_avg_reward = sum(model_no_think_scores) / len(model_no_think_scores)
            think_avg_reward = sum(model_think_scores) / len(model_think_scores)
            
            # Calculate router's average reward at 0.5 threshold
            router_rewards = []
            for i, pred in enumerate(predictions):
                if pred == 'think' and results[i].get('think_score') is not None:
                    router_rewards.append(results[i]['think_score'])
                elif pred == 'no_think' and results[i].get('no_think_score') is not None:
                    router_rewards.append(results[i]['no_think_score'])
            
            router_avg_reward = sum(router_rewards) / len(router_rewards) if router_rewards else 0.0
            
            print(f"\n{'='*80}")
            print(f"Average Reward Analysis")
            print(f"{'='*80}")
            print(f"Always No-Think:  {no_think_avg_reward:.4f}")
            print(f"Always Think:     {think_avg_reward:.4f}")
            print(f"Router (0.5):     {router_avg_reward:.4f}")
            print(f"\nRouter Improvement:")
            print(f"  vs Always No-Think:  {router_avg_reward - no_think_avg_reward:+.4f}")
            print(f"  vs Always Think:     {router_avg_reward - think_avg_reward:+.4f}")
        
        # Diff score statistics
        if 'diff_score' in results[0]:
            diff_scores = [r.get('diff_score') for r in results if r.get('diff_score') is not None]
            if diff_scores:
                avg_diff = sum(diff_scores) / len(diff_scores)
                print(f"\nDiff Score Statistics (think - no_think):")
                print(f"  Average: {avg_diff:.4f}")
                print(f"  Min: {min(diff_scores):.4f}")
                print(f"  Max: {max(diff_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RouteLLM-style confidence-based routers"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="evaluations/routellm/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize evaluator
    evaluator = RouteLLMEvaluator(config)
    
    # Run evaluation
    evaluator.evaluate_dataset()


if __name__ == "__main__":
    main()


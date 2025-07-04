import os
import json
import sys
from datasets import load_dataset

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

# Load the dataset
print("Loading dataset math-500 from Hugging Face...")
ds = load_dataset("huggingfaceh4/math-500")
split = 'test'  # only test split available
print(f"Loaded {len(ds[split])} examples")

# Prepare output directory
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'converted')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'math_500.jsonl')
print(f"Will write converted data to {output_file}")

count = 0
with open(output_file, 'w', encoding='utf-8') as fout:
    for example in ds[split]:
        # Convert to our format, including only standard fields
        record = {
            "original_dataset": "math_500",
            "question": example['problem'],
            "choices": None,
            "choice_index_correct": None,
            "explanation_correct": example['solution'],
            "answer_correct": example['answer']
        }
        # Compute unique_id from the standard fields
        unique_id = utils.compute_unique_id(record)
        record['unique_id'] = unique_id
        
        fout.write(json.dumps(record) + '\n')
        count += 1

print("\n" + "="*50)
print(f"Successfully converted {count} examples and saved to {output_file}")
print("="*50)

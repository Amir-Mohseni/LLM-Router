import os
import json
import sys
from datasets import load_dataset

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

# Load the dataset
ds = load_dataset("huggingfaceh4/math-500")
split = 'test'  # only test split available

# Prepare output directory
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'converted')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'math_500.jsonl')

with open(output_file, 'w', encoding='utf-8') as fout:
    count = 0
    for example in ds[split]:
        # Convert to our format
        record = {
            "original_dataset": "math_500",
            "question": example['problem'],
            "choices": None,
            "choice_index_correct": None,
            "explanation_correct": example['solution'],
            "answer_correct": example['answer'],
            # Keep original fields as metadata
            "subject": example['subject'],
            "level": example['level'],
            "original_unique_id": example['unique_id']
        }
        # Compute unique_id (this will be added to the record)
        unique_id = utils.compute_unique_id(record)
        
        # Add the unique_id and save the record
        record['unique_id'] = unique_id
        fout.write(json.dumps(record) + '\n')
        count += 1

print(f"Converted {count} examples from {split} split and saved to {output_file}")

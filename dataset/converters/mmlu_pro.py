import os
import json
import sys
from datasets import load_dataset

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

print("Loading dataset MMLU Pro from Hugging Face...")
ds = load_dataset("TIGER-Lab/MMLU-Pro")
print(f"Loaded test split: {len(ds['test'])} examples")
print(f"Loaded validation split: {len(ds['validation'])} examples")

# Prepare output directory
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'converted')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'mmlu_pro.jsonl')
print(f"Will write converted data to {output_file}")

records = []
for split in ['test', 'validation']:
    for example in ds[split]:
        # Convert explanation: empty string to None
        explanation = example['cot_content']
        if explanation == "":
            explanation = None
        
        # Create temporary record without unique_id
        temp_record = {
            "original_dataset": "mmlu_pro",
            "question": example['question'],
            "choices": example['options'],
            "choice_index_correct": example['answer_index'],
            "explanation_correct": explanation,
            "answer_correct": example['options'][example['answer_index']]  # set to actual answer text
        }
        
        # Compute unique_id
        unique_id = utils.compute_unique_id(temp_record)
        
        # Create final record
        record = {
            "unique_id": unique_id,
            **temp_record
        }
        records.append(record)

# Write the file
count = len(records)
with open(output_file, 'w', encoding='utf-8') as fout:
    for record in records:
        fout.write(json.dumps(record) + '\n')

# Define validation rules for mmlu_pro dataset
field_rules = {
    "unique_id": {"allow_duplicates": False, "allow_null": False},
    "original_dataset": {"allow_duplicates": True, "allow_null": False},
    "question": {"allow_duplicates": True, "allow_null": False},
    "choices": {"allow_duplicates": True, "allow_null": False},
    "choice_index_correct": {"allow_duplicates": True, "allow_null": False},
    "explanation_correct": {"allow_duplicates": True, "allow_null": True},
    "answer_correct": {"allow_duplicates": True, "allow_null": False}
}

# Validate the dataset
report, num_violations = utils.validate_dataset(records, field_rules)
print(report)

print("\n" + "="*50)
print(f"Successfully converted {count} examples and saved to {output_file}")
print(f"Found {num_violations} rule violations during validation")
print("="*50)

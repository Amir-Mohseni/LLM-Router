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

records = []
for example in ds[split]:
    # Create a temporary record without unique_id for hash computation
    temp_record = {
        "original_dataset": "math_500",
        "question": example['problem'],
        "choices": None,
        "choice_index_correct": None,
        "explanation_correct": example['solution'],
        "answer_correct": example['answer']
    }
    # Compute unique_id from the temporary record
    unique_id = utils.compute_unique_id(temp_record)
    # Create final record with unique_id first
    record = {
        "unique_id": unique_id,
        "original_dataset": temp_record["original_dataset"],
        "question": temp_record["question"],
        "choices": temp_record["choices"],
        "choice_index_correct": temp_record["choice_index_correct"],
        "explanation_correct": temp_record["explanation_correct"],
        "answer_correct": temp_record["answer_correct"]
    }
    records.append(record)

# Write the file
count = len(records)
with open(output_file, 'w', encoding='utf-8') as fout:
    for record in records:
        fout.write(json.dumps(record) + '\n')

# Define validation rules for math_500 dataset
field_rules = {
    "unique_id": {"allow_duplicates": False, "allow_null": False},
    "original_dataset": {"allow_duplicates": True, "allow_null": False},
    "question": {"allow_duplicates": False, "allow_null": False},
    "choices": {"allow_duplicates": True, "allow_null": True},
    "choice_index_correct": {"allow_duplicates": True, "allow_null": True},
    "explanation_correct": {"allow_duplicates": True, "allow_null": False},
    "answer_correct": {"allow_duplicates": True, "allow_null": False}
}

# Validate the dataset
report, num_violations = utils.validate_dataset(records, field_rules)
print(report)

print("\n" + "="*50)
print(f"Successfully converted {count} examples and saved to {output_file}")
print(f"Found {num_violations} rule violations during validation")
print("="*50)

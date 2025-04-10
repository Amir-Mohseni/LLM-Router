import pandas as pd
import hashlib
import json
from typing import Optional


def get_question_id(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()


# Load dataset
splits = {
    "test": "data/test-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["test"])


# Function to normalize each row to the new schema
def normalize_row(row) -> dict:
    return {
        "unique_id": get_question_id(row["question"]),
        "question": row["question"],
        "choices": (row["options"].tolist() if row["options"] is not None else None),
        "answer": (
            row["options"][row["answer_index"]] if row["options"] is not None else None
        ),
        "ground_truth_answer": row["answer_index"],
        "solution": (row["cot_content"] if row["cot_content"] is not None else None),
        "category": row.get("category", None),
    }


# Normalize and write to JSONL
output_path = "mmlu_pro_test.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        structured = normalize_row(row)
        json.dump(structured, f, ensure_ascii=False)
        f.write("\n")

print(f"Serialized dataset saved to {output_path}")

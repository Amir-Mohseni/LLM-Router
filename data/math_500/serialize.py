import pandas as pd
import hashlib
import json
from typing import Optional


def get_question_id(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()


df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)


# Function to normalize each row to the new schema
def normalize_row(row) -> dict:
    return {
        "unique_id": get_question_id(row["problem"]),
        "question": row["problem"],
        "choices": None,
        "answer": (row["answer"] if row["answer"] is not None else None),
        "ground_truth_answer": -1,
        "solution": row["solution"],
        "category": row.get("subject", None),
    }


# Normalize and write to JSONL
output_path = "math_500.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        structured = normalize_row(row)
        json.dump(structured, f, ensure_ascii=False)
        f.write("\n")

print(f"Serialized dataset saved to {output_path}")

import pandas as pd
import hashlib
import json
import sys
from typing import Any, Dict

def make_unique_id(record: Dict[str, Any]) -> str:
    """
    Compute an MD5 hash over the full record.  
    Raises ValueError if 'unique_id' is already present.
    """
    if "unique_id" in record:
        raise ValueError(
            "make_unique_id() expects a record *without* a 'unique_id' key; "
            "remove it before calling or use a fresh dict."
        )

    # Canonical JSON (sorted keys) for deterministic hashing
    canonical = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        default=str
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()

def check_missing_and_empty(df: pd.DataFrame, verbose: bool = False, ignore_columns: list[str] = None) -> tuple[bool, bool]:
    # If columns are specified to be ignored, drop them; ignore errors if a column isn't present.
    if ignore_columns:
        df_check = df.drop(columns=ignore_columns, errors='ignore')
    else:
        df_check = df

    # Check for missing values (None, np.nan, etc.)
    has_missing = df_check.isna().values.any()
    
    # Check for empty strings in cells that contain strings only.
    has_empty = df_check.map(lambda x: isinstance(x, str) and x == "").values.any()

    if verbose:
        messages = []
        if ignore_columns:
            messages.append("NOTE: The following columns are ignored: " + ", ".join(ignore_columns))
        if has_missing:
            messages.append("WARNING: The dataframe has missing values")
        if has_empty:
            messages.append("WARNING: The dataframe has empty string values")
        if not (has_missing or has_empty):
            messages.append("OK: Dataframe has no missing or empty string values")
        print("\n".join(messages))
    
    return has_missing, has_empty

# Load dataset
splits = {
    "test": "data/test-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["test"])

has_missing, has_empty = check_missing_and_empty(df, True, ["cot_content"])
if has_missing or has_empty:
    print("exiting with status code 1...")
    sys.exit(1)

# Function to normalize each row to the new schema
def normalize_row(row) -> dict:
    # Convert options to a list
    choices = row["options"].tolist()
    
    # Retrieve the index of the correct answer
    correct_index = row["answer_index"]
    
    # Get the correct answer based on the index.
    answer_correct = choices[correct_index]

    return {
        "question": row["question"],
        "category": row["category"],
        "choices": choices,
        "choice_index_correct": correct_index,
        "explanation_correct": None,
        "answer_correct": answer_correct,
    }

# Normalize and write to JSONL
output_path = "mmlu_pro_test.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        record = normalize_row(row)

        # compute the unique id
        uid = make_unique_id(record)

        # build a new dict with unique_id at the beginning
        ordered_record_with_id = {"unique_id": uid, **record}

        # write the row to file
        json.dump(ordered_record_with_id, f, ensure_ascii=False)
        f.write("\n")

print(f"Serialized dataset saved to {output_path}")

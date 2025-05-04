import json
import hashlib
import os

def make_unique_id(record):
    if "unique_id" in record:
        raise ValueError("Record already contains 'unique_id'")

    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()

def normalize_record(record):
    choices = record["options"]
    correct_index = record["answer_index"]
    answer_correct = choices[correct_index]

    try:
        model_output = json.loads(record.get("model_outputs", "{}"))
        full_response = model_output.get("response", "")
    except json.JSONDecodeError:
        full_response = ""
    
    # Convert pred letter to index (e.g. "C" -> 2)
    pred_letter = record.get("pred", "").strip().upper()
    try:
        pred_index = ord(pred_letter) - ord("A")
        extracted_answer = choices[pred_index] if 0 <= pred_index < len(choices) else pred_letter
        is_correct = extracted_answer == answer_correct
    except Exception:
        extracted_answer = pred_letter
        is_correct = False

    return {
        "problem": record["question"],
        "is_mcq": True,
        "choices": choices,
        "choice_index_correct": correct_index,
        "explanation_correct": None,
        "answer_correct": answer_correct,
        "category": record.get("category", None),
        "responses": [{
            "full_response": full_response,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct
        }]
    }


# Define file paths
input_path = "/Users/Carrey/Desktop/UM/Year2/MMLU-Pro/eval_results/model_outputs_arx_3_5shots.json"
output_path = input_path.replace(".json", ".jsonl")

# Load input
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure list format
if isinstance(data, dict):
    data = [data]

# Process and write to JSONL
with open(output_path, "w", encoding="utf-8") as out:
    for record in data:
        norm = normalize_record(record)
        uid = make_unique_id(norm)
        final_record = {"unique_id": uid, **norm}
        json.dump(final_record, out, ensure_ascii=False)
        out.write("\n")

print(f"✅ Converted to: {output_path}")

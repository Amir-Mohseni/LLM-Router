import json
from pathlib import Path

def combine_jsonl_files(filepaths, output_file):
    """
    Combine multiple JSONL files, enforcing the following invariants:

    • each record contains a `unique_id`
    • no `unique_id` is repeated across files
    • the reported line count matches the number of unique records written
    """
    seen_ids: set[str] = set()
    line_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for path in map(Path, filepaths):
            with path.open("r", encoding="utf-8") as in_f:
                for raw_line in in_f:
                    # Fast-path skip empty lines / extra whitespace
                    if not raw_line.strip():
                        continue

                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"File “{path}”, malformed JSONL at line {line_count+1}: {e}") from None

                    uid = record.get("unique_id")
                    if uid is None:
                        raise KeyError(f"File “{path}”, line {line_count+1} is missing `unique_id`")

                    if uid in seen_ids:
                        raise ValueError(f"Duplicate unique_id “{uid}” found (first seen earlier, again in “{path}”)")

                    seen_ids.add(uid)
                    out_f.write(raw_line.rstrip("\n") + "\n")
                    line_count += 1

    if line_count != len(seen_ids):
        # This should never happen, but is an extra guard.
        raise AssertionError("Line count mismatch after write; investigate input data integrity.")

    print(
        f"Combined {len(filepaths)} files -> {output_file}\n"
        f"Records written : {line_count}\n"
        f"Unique IDs : {len(seen_ids)}"
    )

# Example usage
if __name__ == "__main__":
    jsonl_files = ["../math_500/math_500.jsonl", "../MMLU-Pro/mmlu_pro_test.jsonl"]
    combine_jsonl_files(jsonl_files, "math_500_and_MMLU_pro.jsonl")

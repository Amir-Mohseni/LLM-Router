import os
import json

def main():
    # Get the script directory and build the converted data path
    script_dir = os.path.dirname(__file__)
    converted_dir = os.path.join(script_dir, '..', 'data', 'converted')
    converted_dir = os.path.abspath(converted_dir)
    
    if not os.path.exists(converted_dir):
        print(f"Error: Converted data directory not found at {converted_dir}")
        print("Make sure to run the dataset converters first")
        return
    
    # Prepare the output file path
    output_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined.jsonl')
    
    # Collect all .jsonl files in the converted directory
    input_files = []
    for filename in os.listdir(converted_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(converted_dir, filename)
            if os.path.isfile(filepath):
                input_files.append(filepath)
    
    if not input_files:
        print(f"No JSONL files found in {converted_dir}")
        return
    
    # Combine all records into one file
    records = []
    for infile_path in input_files:
        with open(infile_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"Invalid JSON record skipped in {infile_path}")
    
    record_count = len(records)
    
    # Write the combined file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for record in records:
            outfile.write(json.dumps(record) + '\n')
    
    print(f"Combined datasets at: {output_path}")
    print(f"Total records: {record_count}")
    print(f"Total files combined: {len(input_files)}")
    
    # Validate the combined dataset
    # Adjust the path to import from dataset.utils
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))
    from utils import validate_dataset
    
    field_rules = {
        "unique_id": {"allow_duplicates": False, "allow_null": False},
        "original_dataset": {"allow_duplicates": True, "allow_null": False},
        "question": {"allow_duplicates": True, "allow_null": False},
        "choices": {"allow_duplicates": True, "allow_null": True},
        "choice_index_correct": {"allow_duplicates": True, "allow_null": True},
        "explanation_correct": {"allow_duplicates": True, "allow_null": True},
        "answer_correct": {"allow_duplicates": True, "allow_null": False}
    }
    report, violations = validate_dataset(records, field_rules)
    
    # Write the report to a file
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Validation report written to: {report_path}")
    print(f"Total rule violations: {violations}")

if __name__ == '__main__':
    main()

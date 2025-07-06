import os
import json

def main():
    # Get the script directory and build the converted data path
    script_dir = os.path.dirname(__file__)
    converted_dir = os.path.join(script_dir, '..', '..', 'data', 'converted')
    converted_dir = os.path.abspath(converted_dir)
    
    # Prepare the output file path
    output_dir = os.path.join(script_dir, '..', '..', 'data')
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
    record_count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for infile_path in input_files:
            with open(infile_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    try:
                        # Validate it's valid JSON before writing
                        json.loads(line)
                        outfile.write(line)
                        record_count += 1
                    except json.JSONDecodeError:
                        print(f"Invalid JSON record skipped in {infile_path}")
    
    print(f"Combined datasets at: {output_path}")
    print(f"Total records: {record_count}")
    print(f"Total files combined: {len(input_files)}")

if __name__ == '__main__':
    main()

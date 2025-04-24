def combine_jsonl_files(filepaths, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for path in filepaths:
            with open(path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(line)
                    
# Example usage:
if __name__ == "__main__":
    # List of JSONL file paths to combine
    jsonl_files = ["../math_500/math_500.jsonl", "../MMLU-Pro/mmlu_pro_test.jsonl"]
    combined_file = "math_500_and_MMLU_pro.jsonl"
    combine_jsonl_files(jsonl_files, combined_file)
    print(f"Files combined into {combined_file}")

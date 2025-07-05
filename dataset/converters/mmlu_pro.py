from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro")

print(ds)
print(ds["test"][0])
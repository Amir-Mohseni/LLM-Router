from datasets import load_dataset

ds = load_dataset("huggingfaceh4/math-500")
print(ds)

print("\nFirst example:")
print(ds['test'][0])

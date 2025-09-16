# Router Evaluation

This directory contains tools for evaluating the reasoning router on datasets.

## Files

- `router_eval.py`: Main evaluation script with batch processing and device optimization
- `config.yaml`: Configuration file for model, dataset, and processing settings
- `results/`: Directory inside evaluations where evaluation results are saved as JSONL files

## Usage

### Basic Usage
```bash
cd evaluations
python router_eval.py
```

### Custom Configuration
```bash
python router_eval.py --config my_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

- **Model**: Change the HuggingFace model ID
- **Dataset**: Specify dataset name, split, and prompt column
- **Processing**: Adjust batch size and max sequence length
- **Device**: Set device preference (auto, cuda, mps, cpu)
- **Output**: Configure output directory and filename

## Output

Results are saved as JSONL files in an organized directory structure inside the evaluations folder:
```
evaluations/results/
├── dataset_name/
│   └── model_name/
│       └── router_predictions.jsonl
```

For example:
- Dataset: `math-ai/aime25` → `aime25`
- Model: `AmirMohseni/reasoning-router-mmbert-small` → `reasoning-router-mmbert-small`
- Output: `evaluations/results/aime25/reasoning-router-mmbert-small/router_predictions.jsonl`

Each line in the JSONL file contains:
- Original dataset fields
- `think_score`: Probability for "think" class
- `no_think_score`: Probability for "no_think" class

## Performance

The script automatically:
- Detects best available device (CUDA > MPS > CPU)
- Processes data in configurable batches for efficiency
- Shows progress bars and summary statistics
- Handles memory management for large datasets

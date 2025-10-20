# Router Evaluation Guide

This directory contains evaluation frameworks for comparing different routing approaches.

## Directory Structure

```
evaluations/
├── router_eval.py           # Classifier-based router (think/no_think)
├── config.yaml              # Config for classifier router
├── routellm/                # RouteLLM competitor module
│   ├── router.py           # Router implementation
│   ├── eval.py             # Evaluation script
│   ├── config.yaml         # Config for RouteLLM
│   └── README.md           # RouteLLM-specific docs
├── results/                # Output directory
└── EVAL_GUIDE.md           # This file
```

---

## 1. Classifier-Based Router (`router_eval.py`)

For **reasoning routers** that predict think/no_think modes using a classifier model.

### Usage:
```bash
cd evaluations
python router_eval.py --config config.yaml
```

### Configuration (`config.yaml`):
- `model.id`: HuggingFace model ID for the classifier
- `labels`: Mapping of class IDs to labels (e.g., 0: "no_think", 1: "think")
- `dataset`: Dataset to evaluate on
- `processing`: Batch size and max length settings
- `device`: Device preference (auto, cuda, mps, cpu)

### Output:
- `results/{dataset}/{model}/router_predictions.jsonl`: Predictions with probabilities for each sample

---

## 2. RouteLLM (`routellm/`)

For **traditional routers** that route between weak/strong models using confidence scores.

### Usage:
```bash
cd evaluations
python -m routellm.eval --config routellm/config.yaml
```

### Configuration (`routellm/config.yaml`):
- `router.type`: RouteLLM router type ("bert", "mf", "sw_ranking", etc.)
- `dataset`: Dataset to evaluate on
- `inference_results`: Model name to load inference results for accuracy/reward comparison

### Output:
- `results/{dataset}/routellm-{type}/router_predictions.jsonl`: Predictions with `no_think_probability` and `think_probability` for each sample
- `results/{dataset}/routellm-{type}/threshold_analysis.json`: Distribution at different thresholds

### Output Format:
```json
{
  "prompt": "...",
  "no_think_probability": 0.407,  // Router confidence for no_think
  "think_probability": 0.593       // Router confidence for think
}
```

### Performance Analysis:
If inference results are provided, the evaluator compares:
- **Always No-Think**: Average accuracy/reward using only no-think mode
- **Always Think**: Average accuracy/reward using only think mode
- **Router**: Average accuracy/reward using router decisions
- Shows improvement over baseline strategies

---

## Adding New Routers

To add a new router models:

1. **Create a new directory**: `evaluations/{router_name}/`
2. **Implement router**: `{router_name}/router.py` with consistent interface
3. **Add evaluator**: `{router_name}/eval.py` with similar structure
4. **Add config**: `{router_name}/config.yaml`
5. **Document**: `{router_name}/README.md`

This modular structure keeps code clean and allows easy comparison between different routing approaches.

---

## Comparing Approaches

Use these evaluators to compare:

1. **Reasoning Router** (think/no_think): Evaluates a single model's capability to use extended reasoning
2. **Traditional Routers** (weak/strong): Evaluates routing between two different models based on query difficulty

Both approaches aim to optimize the performance/cost tradeoff, but use different strategies:
- **Reasoning routers**: Control inference mode of a single model
- **Traditional routers**: Select between different models based on query difficulty


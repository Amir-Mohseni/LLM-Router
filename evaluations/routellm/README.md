# RouteLLM Evaluation

Evaluation module for RouteLLM confidence-based routing (weak/strong model selection).

## Structure

```
routellm/
├── router.py       # Router implementation using routellm library
├── eval.py         # Evaluation script with threshold analysis
├── config.yaml     # Configuration file
└── README.md       # This file
```

## Usage

```bash
# From project root
cd evaluations
python -m routellm.eval --config routellm/config.yaml
```

## Configuration

Edit `config.yaml` to set:
- **Router type**: bert, mf, sw_ranking, etc.
- **Dataset**: HuggingFace dataset to evaluate
- **Inference results**: Model name to load actual inference results (no-think and think modes)

### Inference Results

The evaluator compares router decisions with actual model performance:

**For reward/score-based datasets** (WildChat, Nectar):
```yaml
inference_results:
  model_name: "qwen3-8b"
```

**For accuracy-based datasets** (AIME, HMMT, BRUMO):
```yaml
inference_results:
  model_name: "qwen3-30b"
```

Will automatically look for:
- `results/{dataset}/{model_name}/{model_name}-no-think.jsonl`
- `results/{dataset}/{model_name}/{model_name}-think.jsonl`

And calculate whether the router improves over always using think or no-think modes.

## Output

Results are saved to `evaluations/results/{dataset}/routellm-{type}/`:
- `router_predictions.jsonl`: Router predictions with `no_think_probability` and `think_probability`
- `threshold_analysis.json`: Performance metrics at different thresholds

## Output Format

Each prediction includes:
```json
{
  "prompt": "...",
  "no_think_probability": 0.407,  // Router confidence for no-think
  "think_probability": 0.593       // Router confidence for think
}
```

For datasets with model quality scores:
```json
{
  "think_score": 18.125,           // Model quality score
  "no_think_score": 24.875,        // Model quality score
  "diff_score": -6.75,             // think_score - no_think_score
  "label": "no_think",             // Ground truth label
  "no_think_probability": 0.407,   // Router confidence
  "think_probability": 0.593       // Router confidence
}
```

## Threshold Analysis

The evaluator automatically tests thresholds from 0.1 to 0.9 and reports:
- **Accuracy** (if ground truth available)
- **Distribution**: % routed to think vs no_think mode
- **Best threshold**: Optimal threshold for highest accuracy


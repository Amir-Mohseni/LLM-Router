# Testing Scripts for Data Collection

This directory contains test scripts to demonstrate and validate the data collection components.

## Available Tests

1. **test_jinja_templates.py**: Demonstrates how the Jinja2 templating works for formatting both MCQ and non-MCQ prompts using fixed examples.
   
2. **test_dataset_prompts.py**: Shows how prompts are formatted using actual dataset examples, helping validate the dataset integration.

## Running the Tests

### 1. Testing Jinja Templates

This test demonstrates prompt formatting with fixed examples, without requiring dataset access:

```bash
python -m data_collection.tests.test_jinja_templates
```

**Output**:
- Console display of formatted MCQ and non-MCQ prompts
- Shows exact template rendering with Jinja2

### 2. Testing with Real Dataset

This test loads examples from the configured dataset and formats them:

```bash
python -m data_collection.tests.test_dataset_prompts
```

**Output**:
- Console display of dataset fields and structure
- Example of a formatted MCQ prompt from the dataset
- Example of a formatted non-MCQ prompt from the dataset
- Saves examples to `prompt_examples.json` for reference

## Examples

The MCQ prompt will look like:

```
You are an intelligent assistant designed to provide accurate answers and assist with various tasks.

Answer the following multiple-choice question by analyzing all options carefully.

Instructions:
1. Read the question and all choices thoroughly.
2. Show your step-by-step reasoning for evaluating each option.
3. Present your final answer in the format: <answer>X</answer> where X is the selected option letter.
4. Make sure to wrap your final answer in the <answer> tags.

### Current Problem ###
User: What is the capital of France?
Options:
A. London
B. Paris
C. Rome
D. Berlin
Assistant:
```

The non-MCQ prompt will look like:

```
You are an intelligent assistant designed to provide accurate answers and assist with various tasks.

When presenting mathematical solutions, follow these requirements:

1. **Structured Reasoning**: Show clear step-by-step thinking.
2. **LaTeX Formatting**: Use LaTeX for all mathematical expressions.
3. **Answer Format**: Final answer must be wrapped in <answer> tags with \boxed{} LaTeX.
4. **No Post-Answer Text**: Never add text after the answer block.

### Current Problem ###
User: What is the derivative of f(x) = 3x² + 2x - 5?
Assistant:
```

## Troubleshooting

If you encounter issues with the dataset test:

1. Check that the dataset is properly configured in `data_collection/config.py`
2. Ensure you have the necessary permissions to access the dataset
3. Check that the dataset structure matches what the scripts expect (has 'question', 'choices', etc.) 
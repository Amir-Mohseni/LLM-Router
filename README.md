# LLM Router
## Research Project for Computer Science Bachelor's - Project 2-2

### Research Overview & Roadmap
![LLM Router Flowchart](docs/flowchart.png)
*System flowchart showing the complete research process:*
1. *Query distribution to multiple LLMs*
2. *Answer evaluation using Judge LLM system*
3. *Cost optimization through viable model selection*
4. *BERT-based router training for automated model selection*

### Project Description
This project implements an intelligent routing system for Large Language Models (LLMs) that optimizes for cost while maintaining answer quality. The system uses a BERT-based router trained on historical performance data to direct queries to the most cost-effective LLM capable of answering the query correctly.

### Implementation Steps

1. **Initial Query Distribution**
   - Send the same query to multiple LLMs (e.g., GPT-4o, o1, Deepseek R1, Deepseek V3, Llama 3 8B, Qwen 2.5 7B)
   - Collect and store responses from each model

2. **Answer Quality Assessment**
   - Implement a Judge LLM system
   - Create a database of predefined correct answers
   - Evaluate each model's response with binary classification (Can Answer? Yes/No)
   - Filter out models that provide incorrect or nonsensical answers

3. **Cost Optimization**
   - Among models that can answer correctly, identify the cheapest viable option
   - Create training pairs of [Query, Cheapest Viable LLM]
   - Build a dataset for router training

4. **Router Implementation**
   - Start with an untrained BERT model
   - Fine-tune BERT to predict the optimal (Query, Cheapest Viable LLM) pairs
   - Deploy the trained model as the main router

### Key Components
1. **Multi-Model Query System**
   - Interface with multiple LLM APIs
   - Parallel query processing
   - Response collection and storage

2. **Evaluation System**
   - Judge LLM implementation
   - Answer validation framework
   - Performance tracking

3. **Router Training Pipeline**
   - Data collection and preprocessing
   - BERT fine-tuning system
   - Model evaluation and validation

### Technologies
- Python
- BERT
- Various LLM APIs
- Database for storing results and training data

### Getting Started
(To be added as project develops)

### Contributing
This is a research project for a Computer Science Bachelor's degree. While it's primarily an academic project, feedback and suggestions are welcome through the issues section.

### License
MIT License

---
*This project is part of the Computer Science Bachelor's Program - Project 2-2*

# Hugging Face LLM Chat App

This is a gradio-based chat application that uses Hugging Face's LLM models through their OpenAI-compatible API.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Hugging Face API token as an environment variable:
   ```
   export HF_TOKEN='your_huggingface_token_here'
   ```
   
   You can get your token from your [Hugging Face settings page](https://huggingface.co/settings/tokens).

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser at the URL provided in the terminal (usually http://127.0.0.1:7860)

## Features

- Chat with different LLM models hosted on Hugging Face
- Automatic model selection based on query content
- Conversation history tracking
- Simple, intuitive interface

## Models

The application currently supports:
- Gemma 27B Instruction-tuned
- Llama 3 8B Instruction-tuned

## How to Add More Models

To add more models, modify the `models` dictionary in the `LLMHandler` class in `llm.py`. 
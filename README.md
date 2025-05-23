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

### Architecture
The system uses a polymorphic design with:

1. **BaseLLM** - Abstract base class with common LLM functionality
2. **RemoteLLM** - Concrete implementation for API-based models (OpenAI, etc.)
3. **LocalLLM** - Concrete implementation for local models (vLLM server)
4. **Factory Function** - `create_llm()` that instantiates the appropriate LLM type

This architecture enables:
- Automatic parameter compatibility handling between model types
- Support for both local and remote models through a unified interface
- Clean separation between model types while sharing common functionality

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
- Python 3.10+
- LangChain
- vLLM for local model serving
- Various LLM APIs (OpenAI, etc.)
- Hugging Face Transformers
- Database for storing results and training data

### Getting Started

#### Prerequisites
- Python 3.10+
- Docker (optional, for containerized setup)
- GPU (recommended for local model serving)

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Amir-Mohseni/LLM-Router.git
   cd LLM-Router
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   VLLM_API_KEY=optional_key_for_vllm
   ```

#### Running with Docker (Recommended)
1. Start the services:
   ```bash
   docker-compose up -d
   ```

2. Access the main container:
   ```bash
   docker-compose exec llm-router bash
   ```

3. Run data collection with remote models:
   ```bash
   # Inside the container
   collect my_results.jsonl
   ```

4. Run data collection with local models:
   ```bash
   # Inside the container
   collect my_local_results.jsonl --api_mode local
   ```

### Contributing
This is a research project for a Computer Science Bachelor's degree. While it's primarily an academic project, feedback and suggestions are welcome through the issues section.

### License
MIT License

---
*This project is part of the Computer Science Bachelor's Program - Project 2-2*

# Hugging Face LLM Chat Application

A Gradio-based chat application that intelligently routes user queries to different Large Language Models (LLMs) from Hugging Face using their OpenAI-compatible API.

![Chat Interface Preview](https://i.imgur.com/example-image.png)

## 🌟 Features

- **Multiple Model Support**: Chat with different LLMs hosted on Hugging Face
- **Intelligent Routing**: Automatic model selection based on query content
- **Conversation History**: Full chat history maintained throughout session
- **User-Friendly Interface**: Clean, responsive Gradio UI
- **Model Selection**: Choose models manually or let the router decide
- **Polymorphic Architecture**: Support for both remote API models and local vLLM models

## 🤖 Supported Models

The application currently integrates with:

- **Google/Gemma-3-27b-it**: Google's powerful instruction-tuned model with 27B parameters
- **Meta-llama/Llama-3.2-1B-Instruct**: Meta's efficient instruction-tuned model with 1B parameters
- **Deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B**: Deepseek's powerful thinking and reasoning model with 1.5B parameters

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- A Hugging Face account with API access
- Docker (optional, for containerized setup)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Amir-Mohseni/LLM-Router.git
   cd LLM-Router
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Hugging Face API token:
   ```bash
   export HF_TOKEN='your_huggingface_token_here'
   ```
   
   You can obtain your token from the [Hugging Face settings page](https://huggingface.co/settings/tokens).

## 🚀 Running the Application

Start the application with:

```bash
python app.py
```

Then open your browser at the URL displayed in the terminal (typically http://127.0.0.1:7860).

## 📁 Project Structure

- **app.py**: Main Gradio interface and application entry point
- **data_collection/LLM.py**: Polymorphic LLM interface with support for:
  - `BaseLLM`: Abstract base class
  - `RemoteLLM`: API-based models
  - `LocalLLM`: Local vLLM server models
  - `create_llm()`: Factory function for creating appropriate LLM instances
- **data_collection/run_inference.py**: Script for running inference on datasets with parameter compatibility
- **data_collection/serve_llm.py**: Script for running the local vLLM server
- **router.py**: Smart router that determines which model to use based on content

## 🧠 How the Router Works

The router analyzes user messages to determine the most appropriate model:

- **Creative Content**: For stories, poems, or creative writing → Gemma 27B
- **Complex Questions**: For explanations, analyses, or technical content → Gemma 27B
- **Simple Queries**: For straightforward questions or chat → Llama 3 8B

## 🔧 Customization

### Adding New Models

To add a new model, update the `models` dictionary in `router.py`:

```python
self.models = {
   "google/gemma-3-27b-it": "Gemma 3 27B",
   "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B",
   "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Distill R1 1.5B",
}
```

Then update the routing logic in the `select_model` method.

### Modifying the Interface

The Gradio interface can be customized in `app.py` - refer to the [Gradio documentation](https://www.gradio.app/docs) for more options.

## 📊 Performance Considerations

- The application routes simpler queries to smaller models to balance performance and quality
- For multi-turn conversations, history is limited to the most recent exchanges

## 🐳 Docker Setup

The LLM Router can be run in a Docker container for easy deployment and reproducibility. This approach ensures all dependencies are properly installed and isolated from your system.

### Running with Docker

The application includes a Dockerfile to easily containerize and run the LLM Router.

1. Build the Docker image:
   ```bash
   docker build -t llm-router .
   ```

2. Run the container:
   ```bash
   docker run -p 7860:7860 -e HF_TOKEN=your_huggingface_token_here llm-router
   ```

3. Access the application in your browser at http://localhost:7860

### Environment Variables

- `HF_TOKEN`: Your Hugging Face API token (required)
- You can provide other environment variables using the `-e` flag with `docker run`

### Using Docker Compose

For more advanced setups including GPU support, use the provided docker-compose.yml:

```bash
docker-compose up llm-router
```

This will start the application with the configuration specified in the docker-compose.yml file.

### Persistent Data

To persist data between container runs, you can mount volumes:

```bash
docker run -p 7860:7860 \
  -e HF_TOKEN=your_huggingface_token_here \
  -v $(pwd)/data_collection:/app/data_collection \
  -v $(pwd)/extracted_answers:/app/extracted_answers \
  llm-router
```

## 🧪 Testing

This project includes unit tests and integration tests to ensure the quality and correctness of the data collection and processing components. Tests are implemented using `pytest`.

### Running Standard Tests

These tests cover individual functions and module integration using small data samples. They are generally fast and should be run frequently during development.

1.  **Navigate to the project root directory.**
2.  **Make the test script executable (if you haven't already):**
    ```bash
    chmod +x scripts/run_tests.sh
    ```
3.  **Run the standard tests:**
    ```bash
    ./scripts/run_tests.sh
    ```

### Running Full Dataset Validation

These tests validate the consistency and integrity of the *entire* dataset specified in the configuration. They load all data and can be **very slow** to run.

```bash
./scripts/run_validation.sh
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository. 



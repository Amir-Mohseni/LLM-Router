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
This project implements an intelligent routing system for Large Language Models (LLMs) that optimizes for cost while maintaining answer quality. The system uses a trained router to direct queries to the most cost-effective LLM capable of answering the query correctly, comparing thinking vs non-thinking models across different complexity levels.

### Final Model Evaluation Setup

Based on our research, we evaluated the following models in our final experiments:

#### 🧠 Thinking Models (Advanced Reasoning)
- **Google Gemini 2.5 Pro**: State-of-the-art reasoning with explicit step-by-step thinking
- **Qwen 3 14B**: High-performance thinking model with detailed problem-solving approach

#### ⚡ Non-Thinking Models (Direct Response)
- **Google Gemini 2.0 Flash**: Ultra-fast responses optimized for efficiency
- **Gemma 3 4B**: Lightweight model optimized for quick inference

#### 🎯 Intelligent Router
- **Custom Router**: Trained on Qwen 3 1.7B performance data with RouteLLM integration for automatic model selection

This setup allows us to:
- Compare thinking vs non-thinking capabilities across different model sizes
- Evaluate cost-performance trade-offs between advanced and efficient models
- Test automated routing decisions for optimal model selection

### Architecture
The system uses a polymorphic design with:

1. **BaseLLM** - Abstract base class with common LLM functionality
2. **RemoteLLM** - Concrete implementation for API-based models (OpenAI, Google, etc.)
3. **LocalLLM** - Concrete implementation for local models (vLLM server)
4. **RouteLLMClassifier** - Intelligent routing system using trained classifiers
5. **Factory Function** - `create_llm()` that instantiates the appropriate LLM type

This architecture enables:
- Automatic parameter compatibility handling between model types
- Support for both local and remote models through a unified interface
- Intelligent routing based on query complexity and model capabilities
- Clean separation between model types while sharing common functionality

### Implementation Steps

1. **Multi-Model Query Distribution**
   - Send queries to multiple LLMs with different thinking capabilities
   - Collect responses from both thinking and non-thinking models
   - Store detailed performance and cost metrics

2. **Router Training and Integration**
   - Train router using Qwen 3 1.7B performance data as baseline
   - Integrate with RouteLLM framework for automated model selection
   - Enable automatic strong/weak model routing based on query complexity

3. **Thinking vs Non-Thinking Analysis**
   - Compare reasoning quality between thinking and direct response models
   - Analyze cost-performance trade-offs across different model capabilities
   - Evaluate when explicit reasoning steps improve answer quality

4. **Performance Optimization**
   - Identify optimal routing thresholds for different query types
   - Minimize costs while maintaining answer quality standards
   - Create training datasets for continuous router improvement

### Key Components
1. **Multi-Model Query System**
   - Interface with Google Gemini, Qwen, Gemma, and other model APIs
   - Parallel query processing with thinking/non-thinking modes
   - Response collection and performance tracking

2. **RouteLLM Integration**
   - Pre-trained routing models for query classification
   - Cost-aware model selection capabilities
   - Configurable routing thresholds and model pairs

3. **Evaluation Framework**
   - Comprehensive comparison of thinking vs non-thinking approaches
   - Cost-performance analysis across different model sizes
   - Quality assessment for various query complexity levels

### Getting Started

#### Prerequisites
- Python 3.10+
- Docker (optional, for containerized setup)

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

## 🤖 Final Evaluated Models

The research project evaluated the following models in the final experiments:

### 🧠 Thinking Models (Advanced Reasoning)
- **Google Gemini 2.5 Pro**: State-of-the-art reasoning with explicit step-by-step thinking
- **Qwen 3 14B**: High-performance open-source model with detailed problem-solving capabilities

### ⚡ Non-Thinking Models (Direct Response)  
- **Google Gemini 2.0 Flash**: Ultra-fast responses optimized for efficiency
- **Gemma 3 4B**: Lightweight Google model for quick inference

### 🎯 Intelligent Router
- **Custom Router**: Trained on Qwen 3 1.7B performance data with RouteLLM integration

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

The custom router was trained on Qwen 3 1.7B performance data and integrated with RouteLLM framework:

### Routing Logic:
- **Complex/Technical Queries**: Routes to thinking models (Gemini 2.5 Pro, Qwen 3 14B)
- **Simple/Direct Questions**: Routes to non-thinking models (Gemini 2.0 Flash, Gemma 3 4B)
- **Confidence-Based**: Uses configurable thresholds to balance cost vs. quality

### Usage Example:
```python
from RouteLLM.route_llm_classifier import RouteLLMClassifier

router = RouteLLMClassifier(
    strong_model='google/gemini-2.5-pro-preview',
    weak_model='google/gemini-2.0-flash-001',
    threshold=0.5,
    router_type="bert"
)

# Get routing decision
decision = router.predict_class("Solve this complex math problem...")
# Returns: "strong" or "weak"
```

## 🔧 Customization

### Configuring Model Pairs

To use different model pairs with the router, initialize the `RouteLLMClassifier`:

```python
router = RouteLLMClassifier(
    strong_model='google/gemini-2.5-pro-preview',  # Thinking model
    weak_model='google/gemini-2.0-flash-001',      # Non-thinking model
    threshold=0.5,                                 # Routing threshold
    router_type="bert"                             # Router type
)
```

### Available Model Configurations:
- **Strong Models**: Gemini 2.5 Pro, Qwen 3 14B (thinking capabilities)
- **Weak Models**: Gemini 2.0 Flash, Gemma 3 4B (direct response)

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



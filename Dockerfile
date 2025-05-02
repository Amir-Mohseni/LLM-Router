FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Add pytest for running tests
    pip install --no-cache-dir pytest

# Copy the project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Create directories
RUN mkdir -p data_collection/inference_results extracted_answers

# Create .env file with default configuration
RUN echo "VLLM_API_KEY=dummy-key" > .env

# Set up entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "test" ]; then\n\
    ./scripts/run_tests.sh\n\
elif [ "$1" = "validate" ]; then\n\
    ./scripts/run_validation.sh\n\
elif [ "$1" = "collect" ]; then\n\
    if [ -z "$2" ]; then\n\
        echo "Error: Please provide an output filename"\n\
        echo "Usage: collect <output_filename.jsonl> [--api_mode local|remote]"\n\
        exit 1\n\
    fi\n\
    # Check for optional API mode parameter\n\
    API_MODE="remote"\n\
    if [ "$3" = "--api_mode" ] && [ "$4" = "local" ]; then\n\
        API_MODE="local"\n\
        # If local mode, make sure vLLM server is running\n\
        echo "Using local vLLM mode. Make sure vLLM server is running."\n\
    fi\n\
    python -m data_collection.run_inference --output_file "$2" --api_mode "$API_MODE"\n\
elif [ "$1" = "serve_vllm" ]; then\n\
    # Start the vLLM server with optional model parameter\n\
    MODEL="meta-llama/Llama-3-8B-Instruct"\n\
    if [ -n "$2" ]; then\n\
        MODEL="$2"\n\
    fi\n\
    python -m data_collection.serve_llm --model "$MODEL"\n\
elif [ "$1" = "extract" ]; then\n\
    python -m data_collection.answer_extraction --input "$2"\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (interactive shell if no arguments provided)
CMD ["bash"] 
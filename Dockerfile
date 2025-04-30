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
    python -m data_collection.run_inference --output_file "$2"\n\
elif [ "$1" = "extract" ]; then\n\
    python -m data_collection.answer_extraction --input "$2"\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (interactive shell if no arguments provided)
CMD ["bash"] 
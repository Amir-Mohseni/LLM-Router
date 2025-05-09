FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TOKENIZERS_PARALLELISM=false

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
    pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Make scripts executable (if they exist)
RUN if [ -d "scripts" ]; then chmod +x scripts/*.sh; fi

# Create directories for data persistence
RUN mkdir -p data_collection/inference_results extracted_answers

# Copy example env file if it exists
# Copy .env.example if .env doesn't exist
RUN if [ ! -f ".env" ] && [ -f ".env.example" ]; then cp .env.example .env; fi

# Replace HF_TOKEN in .env with the OpenShift secret if provided
RUN if [ ! -z "$HF_TOKEN" ]; then sed -i "s/^HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env; fi

# Expose the port Gradio will run on
EXPOSE 7860

# Default command to run the app
CMD ["python", "main.py"] 
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

# Copy example env file
COPY .env.example .env

# Default command (interactive shell if no arguments provided)
CMD ["bash"] 
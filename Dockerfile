FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install missing dependencies needed for transformers
RUN pip install --no-cache-dir tiktoken protobuf

# Set longer timeout for Hugging Face downloads
ENV HF_HUB_DOWNLOAD_TIMEOUT=300

# Create model cache directory
RUN mkdir -p /var/lib/model && chmod 777 /var/lib/model

# Use ARG for build-time default values
ARG MODEL_PATH_ARG="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Pre-download the model during build (uses build arg but can be overridden at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_PATH_ARG}', cache_folder='/var/lib/model')"

# Copy application code
COPY src/ /app/src/

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "src/app.py"]
FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the model path and pre-download the model during build
ENV MODEL_PATH="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ENV MODEL_CACHE_DIR="/var/lib/model"
RUN mkdir -p $MODEL_CACHE_DIR && chmod 777 $MODEL_CACHE_DIR
# Pre-download the model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$MODEL_PATH', cache_folder='$MODEL_CACHE_DIR')"

# Copy application code
COPY src/ /app/src/

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "src/app.py"]
FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/

# Create model cache directory and ensure proper permissions
RUN mkdir -p /var/lib/model && chmod 777 /var/lib/model

# Expose port (Dokku will override this)
EXPOSE 8000

# Start the application
CMD ["python", "src/app.py"]
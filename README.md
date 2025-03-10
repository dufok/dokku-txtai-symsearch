# TxtAI Service for Discourse.io

## Overview

TxtAI Service is a dedicated microservice that powers the text embeddings and semantic search functionality for Discourse.io. It leverages the [txtai](https://github.com/neuml/txtai) library to provide efficient vector-based search across articles and authors.

This service offloads compute-intensive ML operations from the main backend, improving overall system performance and resource utilization.

## Features

- **Multilingual Text Embeddings**: Uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` for high-quality embeddings
- **Document Indexing**: Efficient indexing of single documents or batches
- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **Persistent Model Cache**: Model files are stored in persistent storage to avoid redownloading
- **Redis Integration**: Optional result caching using Redis
- **Health Monitoring**: Service status endpoints for monitoring

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/info` | GET | Get service status and model information |
| `/index` | POST | Index a single document |
| `/bulk-index` | POST | Index multiple documents at once |
| `/search` | POST | Search for relevant documents |

### Example Requests

```bash
# Check service status
curl -X GET http://localhost:8000/info

# Index a document
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"id":"123", "text":"Sample article text to index"}'

# Search for content
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"text":"search query", "limit":10, "offset":0}'
```

## Dokku setup
```
# Create the app
dokku apps:create txtai

# Configure environment variables
dokku config:set txtai REDIS_URL="redis://:password@dokku-redis-discoursio-redis:6379"
dokku config:set txtai MODEL_CACHE_DIR="/var/lib/model"  
dokku config:set txtai MODEL_PATH="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Create persistent storage for model cache
dokku storage:ensure-directory txtai
dokku storage:mount txtai /var/lib/dokku/data/storage/txtai:/var/lib/model

# Set domain (optional)
dokku domains:set txtai txtai.yourdomain.com
```
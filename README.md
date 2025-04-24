# TxtAI Hybrid Search Service for Discourse.io

## Overview

TxtAI Hybrid Search Service is a dedicated microservice that powers text embeddings and advanced search functionality for Discourse.io. Built on the [txtai](https://github.com/neuml/txtai) library, this service now leverages PostgreSQL (with the pgvector extension) to store document content and vector embeddings persistently. In addition to semantic search, it supports traditional keyword (syntax) search and can combine both using a weighted hybrid approach. This design minimizes on-demand computation and significantly reduces system load, ensuring efficient and secure user experiences.

## Features

- **Multilingual Text Embeddings**: Uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` for high-quality embeddings.
- **Document Indexing**: Incremental indexing (via upsert) allows adding or updating documents without reprocessing the entire database.
- **Hybrid Search**: Combines semantic (vector-based) search with full-text (syntax/BM25) search using weighted scoring.
- **Persistent Storage with PostgreSQL**: Uses PostgreSQL (via DATABASE_URL) with pgvector to securely store document content and embeddings.
- **Redis Integration (Optional)**: Supports result caching using Redis.
- **Persistent Model Cache**: Model files are stored in a persistent directory to avoid redownloading.
- **Health Monitoring**: Provides endpoints for service status and model information.

## API Endpoints

| Endpoint             | Method | Description                                                       |
|----------------------|--------|-------------------------------------------------------------------|
| `/info`              | GET    | Get service status and current configuration.                     |
| `/health`            | GET    | Comprehensive health check of database, model, and required tables.|
| `/dbtest`            | GET    | Test database connection and configuration (diagnostic endpoint). |
| `/index-status`      | GET    | Get detailed statistics about the current index state.            |
| **Body Content**     |        |                                                                   |
| `/index-body`        | POST   | Index (or update) a single body document using semantic search.   |
| `/bulk-index-bodies` | POST   | Index multiple body documents in a single request.                |
| **Title Content**    |        |                                                                   |
| `/index-title`       | POST   | Index (or update) a single title using fuzzy search.              |
| `/bulk-index-titles` | POST   | Index multiple titles in a single request.                        |
| **Author Content**   |        |                                                                   |
| `/index-author`      | POST   | Index (or update) a single author using hybrid search.            |
| `/bulk-index-authors`| POST   | Index multiple authors in a single request.                       |
| **Search**           |        |                                                                   |
| `/search-combined`   | POST   | Perform prioritized search with exact titles, all-terms titles, and body semantic matches. |
| `/search-author`     | POST   | Search authors using hybrid approach (semantic + fuzzy).          |
| **Maintenance**      |        |                                                                   |
| `/reset-connection`  | POST   | Reset all database connections and reinitialize the embeddings object. |
| `/verify-docs`       | POST   | Verify which documents exist in the index from a provided list of IDs. |
| `/delete-all`        | POST   | Delete all documents and embeddings from the database (for testing only). |

> **Legacy endpoints**: For backward compatibility, these endpoints are still available but reference the new specialized endpoints:
> - `/index` (POST): Maps to `/index-body`
> - `/bulk-index` (POST): Maps to `/bulk-index-bodies`
> - `/search` (POST): Maps to body search functionality

### Example Requests for Specialized Endpoints

```bash
# Index a body document
curl -X POST http://localhost:8000/index-body \
  -H "Content-Type: application/json" \
  -d '{"id": "123", "body": "Sample article text to index"}'

# Index a title
curl -X POST http://localhost:8000/index-title \
  -H "Content-Type: application/json" \
  -d '{"id": "123", "title": "Sample Article Title"}'

# Index an author
curl -X POST http://localhost:8000/index-author \
  -H "Content-Type: application/json" \
  -d '{"id": "author1", "name": "Jane Doe", "bio": "Award-winning author"}'

# Combined search (with prioritized results)
curl -X POST http://localhost:8000/search-combined \
  -H "Content-Type: application/json" \
  -d '{"text": "search query", "limit": 10}'

# Search authors
curl -X POST http://localhost:8000/search-author \
  -H "Content-Type: application/json" \
  -d '{"text": "author name", "limit": 10}'

# Bulk index titles
curl -X POST http://localhost:8000/bulk-index-titles \
  -H "Content-Type: application/json" \
  -d '[{"id": "123", "title": "First title"}, {"id": "124", "title": "Second title"}]'
```

# Verify specific documents exist in the index
curl -X POST http://localhost:8000/verify-docs \
  -H "Content-Type: application/json" \
  -d '{"doc_ids": ["123", "456", "789"]}'

# Get current index statistics
curl -X GET http://localhost:8000/index-status

# Synchronize index with source of truth (with optional document fetching)
curl -X POST http://localhost:8000/sync-index \
  -H "Content-Type: application/json" \
  -d '{
    "doc_ids": ["123", "456", "789"],
    "fetch_callback_url": "http://your-backend/api/get-document-content"
  }'

> Note: The service automatically checks for and initializes the pgvector extension and required database tables on startup, so manual database setup is no longer required.

#### Dokku setup

# Specific points:

  - We are need Postgres version with pgvector

```bash
  dokku apps:create txtai
  dokku postgres:create txtai-db --image "pgvector/pgvector" --image-version "pg16"
```
  - Connect to DB and activate vector

```bash
  dokku postgres:connect txtai-db

  txtxai_db=# CREATE EXTENSION vector;
  txtxai_db=# CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

  - Link with app

```bash
  dokku postgres:link txtai-db txtai
```

  - Config app

```bash
  dokku config:set txtai MODEL_CACHE_DIR="/var/lib/model"
  dokku config:set txtai MODEL_PATH="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  dokku storage:ensure-directory txtai
  dokku storage:mount txtai /var/lib/dokku/data/storage/txtai:/var/lib/model
  dokku network:create core-searchtxtai-bridge
```

  - Config one netwrok for backend and frontend

```bash
  dokku network:set BACKEND attach-post-create core-searchtxtai-bridge
  dokku network:set txtai attach-post-create core-searchtxtai-bridge
  dokku config:set BACKEND TXTAI_SERVICE_URL=http://txtai.web.1:8000
```

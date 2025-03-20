import os

# ------------------------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------------------------
# DATABASE_URL: PostgreSQL connection string, e.g.
#   postgres://postgres:YOUR_PASSWORD@dokku-postgres-txt-ai-db:5432/txt_ai_db
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable is not set")

# REDIS_URL: Redis connection string (optional, for caching)
REDIS_URL = os.getenv("REDIS_URL")

# MODEL_CACHE_DIR: Directory for persistent model caching
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/var/lib/model")

# MODEL_PATH: Model path (HuggingFace model ID or local path)
MODEL_PATH = os.getenv("MODEL_PATH", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Optional hybrid search weights. Adjust these to favor full-text (syntax) or semantic search.
LEX_WEIGHT = float(os.getenv("LEX_WEIGHT", "0.5"))
SEM_WEIGHT = float(os.getenv("SEM_WEIGHT", "0.5"))

# ------------------------------------------------------------------------------
# Txtai Configuration
# ------------------------------------------------------------------------------
# This configuration dictionary is used to initialize the txtai Embeddings instance.
# It specifies the vector model, caching options, and tells txtai to use PostgreSQL
# for storing document content and embeddings via the pgvector backend.
config = {
    "path": MODEL_PATH,
    "cache": MODEL_CACHE_DIR,        # directory to persist model files
    "content": DATABASE_URL,         # store document content in PostgreSQL
    "backend": "pgvector",           # use PostgreSQL with pgvector extension for vector search
    "pgvector": {
        "url": DATABASE_URL         # PostgreSQL connection for pgvector operations
    },
    "scoring": {
        "method": "bm25",          # enable syntax (full-text) search via BM25
        "terms": True
    },
    # Hybrid search settings: weights for combining full-text and semantic search.
    "hybrid": {
        "lex_weight": LEX_WEIGHT,
        "sem_weight": SEM_WEIGHT
    },
    # Flag to indicate if incremental indexing (upsert) is enabled.
    "incremental": True
}

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def get_config():
    """
    Returns the txtai configuration dictionary.
    """
    return config

if __name__ == "__main__":
    # For testing: print the configuration as formatted JSON.
    import json
    print(json.dumps(config, indent=2))

import os
from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel
from txtai.embeddings import Embeddings
from sqlalchemy import create_engine, text
import numpy as np

app = FastAPI(title="TxtAI Service")

logger = logging.getLogger("txtai")
logger.setLevel(logging.INFO) 

# Load environment variables (dokku sets these)
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/var/lib/model")
MODEL_PATH = os.getenv("MODEL_PATH")
REDIS_URL = os.getenv("REDIS_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL is not set")
else:
    # Convert postgres:// to postgresql:// for SQLAlchemy compatibility
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Build txtai configuration to use Postgres for content storage and pgvector for vector search.
config = {
    "path": MODEL_PATH,
    "cache": MODEL_CACHE_DIR,          # persistent model cache
    "content": DATABASE_URL,           # store document content in PostgreSQL
    "backend": "pgvector",             # use pgvector as ANN backend
    "pgvector": {"url": DATABASE_URL}, # pgvector connection settings
    "scoring": {"method": "bm25", "terms": True}  # enable syntax search (BM25)
}

# Initialize txtai embeddings instance with better error handling
try:
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH environment variable is not set")
    
    embeddings = Embeddings(config)
    print(f"Model {MODEL_PATH} initialized successfully")
except Exception as e:
    print(f"Error initializing embeddings model: {str(e)}")
    raise  # Re-raise to prevent startup with a broken model

# Set up SQLAlchemy engine for direct PostgreSQL queries.
engine = create_engine(DATABASE_URL)

# Define Pydantic models for API request bodies.
class Document(BaseModel):
    id: str
    text: str

class SearchRequest(BaseModel):
    text: str
    limit: int = 10

@app.get("/info")
def info():
    """
    Returns the current configuration and service status.
    """
    return {"config": config, "status": "Txtai service running"}

@app.post("/index")
def index_document(doc: Document):
    """
    Incrementally index a single document using upsert.
    """
    try:
        # Use upsert so that only new data is indexed (or updated if already present)
        embeddings.upsert([(doc.id, doc.text, None)])
        return {"message": f"Document {doc.id} indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-index")
def bulk_index(docs: list[Document]):
    """
    Incrementally index multiple documents.
    """
    try:
        data = [(doc.id, doc.text, None) for doc in docs]
        embeddings.upsert(data)
        return {"message": f"{len(docs)} documents indexed"}
    except Exception as e:
        logger.error(f"Bulk indexing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def get_query_embedding(query: str) -> list[float]:
    """
    Generate the query embedding using txtai.
    """
    vector = embeddings.transform(query)
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)

@app.post("/search")
def search(req: SearchRequest):
    """
    Performs a hybrid search combining:
      - Full-text search (syntax via BM25) on the tsvector column.
      - Semantic search (via pgvector) on stored embeddings.
      
    The results from each method are combined with weighted scores.
    """
    query = req.text
    limit = req.limit

    # Compute the query embedding for semantic search.
    query_embedding = get_query_embedding(query)

    # Define weights for hybrid search (adjustable based on query context)
    lex_weight = 0.5  # weight for full-text (syntax) search
    sem_weight = 0.5  # weight for semantic (vector) search

    # Hybrid search SQL:
    # 1. The "full_text" CTE finds documents matching the query using full-text search.
    # 2. The "semantic" CTE retrieves documents by computing cosine similarity via pgvector.
    # 3. We combine the scores using a weighted sum.
    hybrid_sql = text("""
        WITH 
            full_text AS (
                SELECT id, 
                       ts_rank_cd(content_tsv, websearch_to_tsquery('english', :query)) AS lex_score
                FROM documents
                WHERE content_tsv @@ websearch_to_tsquery('english', :query)
                ORDER BY lex_score DESC
                LIMIT 50
            ),
            semantic AS (
                SELECT id,
                       (1 - (embedding <#> :query_embedding)) AS sem_score
                FROM documents
                ORDER BY embedding <#> :query_embedding ASC
                LIMIT 50
            )
        SELECT d.id, d.content,
               (COALESCE(ft.lex_score, 0) * :lex_weight +
                COALESCE(sm.sem_score, 0) * :sem_weight) AS combined_score
        FROM full_text ft
        FULL OUTER JOIN semantic sm ON ft.id = sm.id
        JOIN documents d ON d.id = COALESCE(ft.id, sm.id)
        ORDER BY combined_score DESC
        LIMIT :limit;
    """)

    with engine.connect() as conn:
        results = conn.execute(hybrid_sql, {
            "query": query,
            "query_embedding": query_embedding,
            "lex_weight": lex_weight,
            "sem_weight": sem_weight,
            "limit": limit
        })
        rows = [dict(row) for row in results]

    return {"results": rows}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

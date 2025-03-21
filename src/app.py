import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from txtai.embeddings import Embeddings
from sqlalchemy import create_engine, text
import numpy as np
import logging
import sys

app = FastAPI(title="TxtAI Service")

# Configure root logger to ensure errors show up
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
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
    "cache": MODEL_CACHE_DIR, 
    "content": DATABASE_URL,
    "backend": "pgvector",
    "pgvector": {
        "url": DATABASE_URL,
        "table": "embeddings",     # Tell txtai to use this table for vectors
        "content": "sections"      # Tell txtai to use this table for content
    },
    "scoring": {"method": "bm25", "terms": True}
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
        # Add print statement for direct console output
        print(f"Processing bulk index request with {len(docs)} documents")
        
        # Add validation and document size reporting
        doc_sizes = [len(doc.text) for doc in docs]
        max_size = max(doc_sizes) if doc_sizes else 0
        avg_size = sum(doc_sizes)/len(doc_sizes) if doc_sizes else 0
        print(f"Document stats: max_size={max_size}, avg_size={avg_size:.1f}")
        
        data = [(doc.id, doc.text, None) for doc in docs]
        embeddings.upsert(data)
        return {"message": f"{len(docs)} documents indexed"}
    except Exception as e:
        # Use print for guaranteed output in logs
        print(f"CRITICAL ERROR in bulk_index: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Still use logger but it might not show up
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

@app.post("/setup")
def setup_database():
    """Initialize the database schema from scratch."""
    global embeddings
    try:
        print("Setting up database schema...")
        
        # Force close any existing connections
        try:
            embeddings.close()
        except:
            pass
            
        # Recreate the embeddings instance with schema creation
        config_with_setup = config.copy()
        config_with_setup["create"] = True  # Force schema creation
        
        embeddings = Embeddings(config_with_setup)
        
        print("Database schema created successfully")
        return {"message": "Database schema created successfully"}
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize")
def initialize_db():
    """Reset and reinitialize the database schema."""
    global embeddings
    try:
        print("Checking database schema...")
        
        with engine.connect() as conn:
            # First get the actual sequence names
            sequences = conn.execute(text("SELECT sequencename FROM pg_sequences WHERE schemaname = 'public'")).fetchall()
            tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
            
            print(f"Found sequences: {sequences}")
            print(f"Found tables: {tables}")
            
            # Only try to reset if we find sequences
            if sequences:
                for seq in sequences:
                    seq_name = seq[0]
                    print(f"Resetting sequence {seq_name}")
                    conn.execute(text(f"ALTER SEQUENCE {seq_name} RESTART WITH 1"))
            
            conn.commit()
            
        return {"message": "Database initialized successfully", "sequences": [s[0] for s in sequences], "tables": [t[0] for t in tables]}
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/dbtest")
def db_test():
    """Test database connection and print URL."""
    try:
        print(f"Testing database connection with URL: {DATABASE_URL}")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            print(f"Connection successful: {result}")
            # Try creating a test table directly
            conn.execute(text("CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name TEXT)"))
            conn.commit()
            print("Test table created")
            return {"status": "success", "url": DATABASE_URL.replace(DATABASE_URL.split('@')[0], "***")}
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.get("/check-pgvector")
def check_pgvector():
    """Check if pgvector extension is available."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
            )).fetchone()
            
            if result:
                return {"status": "success", "pgvector": True, "version": result[1]}
            else:
                # Try to create the extension
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    return {"status": "success", "pgvector": "installed", "message": "Extension created"}
                except Exception as ext_error:
                    return {"status": "error", "pgvector": False, "message": str(ext_error)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/setup-manual")
def setup_manual():
    """Manually create txtai schema."""
    try:
        with engine.connect() as conn:
            # Create sections table that txtai uses
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sections (
                    indexid SERIAL PRIMARY KEY,
                    id TEXT NOT NULL,
                    text TEXT,
                    tags TEXT,
                    entry TIMESTAMP WITH TIME ZONE
                )
            """))
            
            # Try to create vector table if pgvector is available
            try:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id TEXT PRIMARY KEY,
                        embedding VECTOR(384)
                    )
                """))
            except Exception as vector_error:
                print(f"Vector table creation error: {str(vector_error)}")
                # Continue anyway
            
            conn.commit()
            return {"status": "success", "message": "Tables created manually"}
    except Exception as e:
        print(f"Manual setup error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    
@app.get("/health")
def health_check():
    """Check if the service is healthy with working database and model."""
    try:
        # Check database
        with engine.connect() as conn:
            db_ok = conn.execute(text("SELECT 1")).fetchone() is not None
        
        # Check model by testing a simple embedding operation
        model_ok = False
        try:
            # Try to embed a simple test string
            test_vector = embeddings.transform("test")
            model_ok = test_vector is not None and len(test_vector) > 0
        except Exception as model_error:
            print(f"Model health check failed: {str(model_error)}")
            model_ok = False
        
        # Check if tables exist
        tables_exist = False
        with engine.connect() as conn:
            tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
            tables_exist = len(tables) > 0
            
        return {
            "status": "healthy" if (db_ok and model_ok and tables_exist) else "unhealthy",
            "database": "connected" if db_ok else "disconnected",
            "model": "loaded" if model_ok else "not_loaded",
            "tables": "exist" if tables_exist else "missing"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
    
@app.post("/clean-reset")
def clean_reset():
    """Complete reset: truncate tables, reset sequences, and clear txtai state."""
    global embeddings
    try:
        with engine.connect() as conn:
            # 1. Truncate all tables to remove data
            conn.execute(text("TRUNCATE sections, embeddings RESTART IDENTITY CASCADE"))
            
            # 2. Reset the sequence explicitly
            conn.execute(text("ALTER SEQUENCE sections_indexid_seq RESTART WITH 1"))
            
            # 3. Close and recreate the embeddings instance to clear state
            try:
                embeddings.close()
            except:
                pass
                
            conn.commit()
            
        # Recreate the txtai embeddings with clean state
        embeddings = Embeddings(config)
            
        return {
            "status": "success", 
            "message": "Database completely reset - tables truncated and sequences reset"
        }
    except Exception as e:
        print(f"Clean reset error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

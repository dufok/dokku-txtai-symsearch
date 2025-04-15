import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from txtai.embeddings import Embeddings
from sqlalchemy import create_engine, text
import numpy as np
import logging
import sys
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="TxtAI Service")

class VerifyRequest(BaseModel):
    doc_ids: List[str]

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


# Set up SQLAlchemy engine for direct PostgreSQL queries.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Check and initialize pgvector extension on startup
print("Checking PostgreSQL pgvector extension...")
try:
    with engine.connect() as conn:
        # Check if vector extension exists
        result = conn.execute(text(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        )).fetchone()
        
        if result:
            print(f"pgvector extension found with version {result[1]}")
        else:
            print("pgvector extension not found, attempting to create it...")
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                print("Successfully created pgvector extension")
            except Exception as ext_error:
                print(f"WARNING: Could not create pgvector extension: {str(ext_error)}")
                print("Vector operations may not work correctly!")
except Exception as e:
    print(f"Error checking pgvector extension: {str(e)}")
    print("Continuing startup, but vector operations may fail")

# Check if database has required schema and initialize if needed
print("Checking if database has required tables...")
try:
    with engine.connect() as conn:
        # Check if essential tables exist
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('sections', 'embeddings', 'documents')
        """)).fetchall()
        
        existing_tables = [row[0] for row in result]
        print(f"Found tables: {existing_tables}")
        
        if len(existing_tables) < 3:
            print("Database is missing essential tables. Initializing schema...")
            
            # Create sections table that txtai uses if it doesn't exist
            if 'sections' not in existing_tables:
                print("Creating sections table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sections (
                        indexid SERIAL PRIMARY KEY,
                        id TEXT NOT NULL,
                        text TEXT,
                        tags TEXT,
                        entry TIMESTAMP WITH TIME ZONE
                    )
                """))
            
            # Create embeddings table with indexid column that txtai expects
            if 'embeddings' not in existing_tables:
                print("Creating embeddings table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        indexid SERIAL PRIMARY KEY,
                        id TEXT UNIQUE NOT NULL,
                        embedding VECTOR(384)
                    )
                """))
            
            # Create documents table with full-text search capabilities
            if 'documents' not in existing_tables:
                print("Creating documents table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR(384),
                        content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                    )
                """))
                
                # Create index for full-text search
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS documents_content_tsv_idx ON documents USING GIN (content_tsv)
                """))
                
                # Create index for vector similarity search
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
                    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
                """))
            
            conn.commit()
            print("Database schema initialized successfully")
        else:
            print("All required tables already exist")

except Exception as e:
    print(f"Error checking/initializing database schema: {str(e)}")
    import traceback
    print(traceback.format_exc())
    print("Continuing startup, but database operations may fail")
    # Make sure to roll back any failed transaction
    try:
        if 'conn' in locals():
            conn.rollback()
    except:
        pass

# Build txtai configuration to use Postgres for content storage and pgvector for vector search.
config = {
    "path": MODEL_PATH,
    "cache": MODEL_CACHE_DIR, 
    "content": DATABASE_URL,
    "backend": "pgvector",
    "pgvector": {
        "url": DATABASE_URL,
        "table": "documents",      # Use documents table for vectors
        "content": "documents",    # Use documents table for content
        "column": "embedding",     # Specify embedding column name
        "content_column": "content" # Specify content column name
    },
    "scoring": {"method": "bm25", "terms": True},
    "incremental": True
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
    Returns the current configuration and service status with index statistics.
    """
    try:
        # Get document count
        doc_count = 0
        with engine.connect() as conn:
            doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        
        return {
            "config": config, 
            "status": "Txtai service running",
            "index_stats": {
                "document_count": doc_count,
                "model": MODEL_PATH,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"Error getting info: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/index")
def index_document(doc: Document):
    """
    Incrementally index a single document using upsert.
    """
    try:
        # Use upsert so that only new data is indexed (or updated if already present)
        embeddings.upsert([(doc.id, doc.text, None)])
        
        # Also insert into the documents table directly
        with engine.connect() as conn:
            # Get the embedding vector for the document
            embedding = embeddings.transform(doc.text)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Insert into documents table
            stmt = text("""
                INSERT INTO documents (id, content, embedding)
                VALUES (:id, :content, CAST(:embedding AS vector(384)))
                ON CONFLICT (id) DO UPDATE
                SET content = :content, embedding = CAST(:embedding AS vector(384))
            """)
            
            conn.execute(stmt, {
                "id": doc.id,
                "content": doc.text,
                "embedding": embedding
            })
            conn.commit()
        
        return {"message": f"Document {doc.id} indexed"}
    except Exception as e:
        import traceback
        print(f"Indexing error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-index")
def bulk_index(docs: list[Document]):
    """
    Incrementally index multiple documents with improved batching to prevent errors.
    """
    global embeddings, engine
    
    try:
        print(f"Processing bulk index request with {len(docs)} documents")
        
        # Document statistics for debugging
        doc_sizes = [len(doc.text) for doc in docs]
        max_size = max(doc_sizes) if doc_sizes else 0
        avg_size = sum(doc_sizes)/len(doc_sizes) if doc_sizes else 0
        print(f"Document stats: max_size={max_size}, avg_size={avg_size:.1f}")
        
        # Prepare all data
        data = [(doc.id, doc.text, None) for doc in docs]
        
        # Process in small batches to prevent dictionary size issues
        batch_size = 3  # Very small batches to prevent dictionary mutation errors
        success_count = 0
        failed_docs = []
        
        # First pass - process documents in small batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            try:
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                embeddings.upsert(batch)

                # Also update the documents table
                with engine.connect() as conn:
                    for doc_id, doc_text, _ in batch:
                        # Get embedding for this document
                        embedding = embeddings.transform(doc_text)
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        
                        # Insert into documents table
                        stmt = text("""
                            INSERT INTO documents (id, content, embedding)
                            VALUES (:id, :content, CAST(:embedding AS vector(384)))
                            ON CONFLICT (id) DO UPDATE
                            SET content = :content, embedding = CAST(:embedding AS vector(384))
                        """)
                        
                        conn.execute(stmt, {
                            "id": doc_id,
                            "content": doc_text,
                            "embedding": embedding
                        })
                    conn.commit()

                success_count += len(batch)
                # Small sleep between batches to prevent overloading the database
                if i + batch_size < len(data):
                    import time
                    time.sleep(0.2)
            except Exception as batch_error:
                print(f"Batch {batch_num} failed: {str(batch_error)}")
                # Track failed documents to retry individually
                failed_docs.extend(batch)
        
        # If we had any failures, try to reset connections and retry one by one
        if failed_docs:
            print(f"First pass: {success_count} of {len(docs)} documents indexed successfully")
            print(f"Retrying {len(failed_docs)} failed documents individually with fresh connections")
            
            # Reset connections
            engine.dispose()
            import time
            time.sleep(1)
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            embeddings = Embeddings(config)
            print("Reset connections for individual retries")
            
            # Retry each failed document individually with more robust error handling
            for doc_data in failed_docs:
                try:
                    print(f"Retrying document {doc_data[0]}")
                    # Index just this one document
                    embeddings.upsert([doc_data])
                    success_count += 1
                    # Small delay between individual retries
                    time.sleep(0.3)
                except Exception as retry_error:
                    print(f"Retry failed for document {doc_data[0]}: {str(retry_error)}")
        
        # Report final results
        if success_count == len(docs):
            return {"message": f"All {len(docs)} documents indexed successfully"}
        else:
            return {
                "message": f"{success_count} of {len(docs)} documents indexed",
                "status": "partial_success" if success_count > 0 else "failure"
            }
            
    except Exception as e:
        # Use print for guaranteed output in logs
        print(f"CRITICAL ERROR in bulk_index: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Still use logger but it might not show up
        logger.error(f"Bulk indexing error: {str(e)}", exc_info=True)
        
        # This ensures we have clean DB connections for future requests
        try:
            engine.dispose()
            # Add a delay before recreation
            import time
            time.sleep(1)
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            embeddings = Embeddings(config)
            print("Reset connections after error")
        except Exception as reset_error:
            print(f"Failed to reset connections: {str(reset_error)}")
        
        raise HTTPException(status_code=500, detail=str(e))

# Add the missing function for query embedding generation
def get_query_embedding(query_text):
    """
    Generate embedding vector for search queries.
    """
    try:
        # Use the embeddings model to transform the query text into a vector
        query_vector = embeddings.transform(query_text)
        
        # Convert to numpy array if needed (pgvector expects numpy arrays)
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
            
        return query_vector
    except Exception as e:
        print(f"Error generating query embedding: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query embedding generation failed: {str(e)}")

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

    # Convert NumPy array to a list format PostgreSQL can handle
    if isinstance(query_embedding, np.ndarray):
        # For pgvector compatibility
        query_embedding = query_embedding.tolist()

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
                       (1 - (embedding <#> CAST(:query_embedding AS vector(384)))) AS sem_score
                FROM documents
                ORDER BY embedding <#> CAST(:query_embedding AS vector(384)) ASC
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
        rows = []
        for row in results:
            # Access by column name instead of trying to convert directly to dict
            rows.append({
                "id": row.id,
                "content": row.content,
                "score": float(row.combined_score)  # Convert Decimal to float for JSON serialization
            })

    return {"results": rows}
    
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
    
@app.post("/reset-connection")
def reset_connection():
    """Complete reset of database connections and embeddings object."""
    global embeddings, engine
    
    try:
        # First, dispose of all connections in the pool
        engine.dispose()
        
        # Recreate the engine
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
        # Recreate embeddings object with fresh connections
        embeddings = Embeddings(config)
        
        return {
            "status": "success",
            "message": "Database connections reset and embeddings object recreated"
        }
    except Exception as e:
        print(f"Connection reset error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    
@app.post("/verify-docs")
def verify_documents(req: VerifyRequest):
    """
    Verify which documents from the provided list exist in the index.
    Returns lists of missing and existing document IDs.
    """
    try:
        with engine.connect() as conn:
            # Find which IDs exist in the database
            placeholders = ", ".join([f"'{id}'" for id in req.doc_ids])
            query = f"SELECT id FROM documents WHERE id IN ({placeholders})"
            result = conn.execute(text(query))
            
            existing_ids = [row[0] for row in result]
            missing_ids = [id for id in req.doc_ids if id not in existing_ids]
            
            return {
                "total_requested": len(req.doc_ids),
                "existing": existing_ids,
                "missing": missing_ids,
                "exists_count": len(existing_ids),
                "missing_count": len(missing_ids)
            }
    except Exception as e:
        print(f"Error verifying documents: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    
@app.get("/index-status")
def index_status():
    """Return detailed statistics about the current index state."""
    try:
        print("Index status endpoint called - checking statistics")
        with engine.connect() as conn:
            # Get basic counts
            doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
            
            # Check for null embeddings
            null_embed_query = text("""
                SELECT id FROM documents 
                WHERE embedding IS NULL
                LIMIT 10
            """)
            null_embeddings = [row[0] for row in conn.execute(null_embed_query)]
            
            # Get sample of document IDs
            sample_ids = [row[0] for row in conn.execute(text("SELECT id FROM documents ORDER BY id LIMIT 5"))]
            
            # Check ID ranges
            id_range = conn.execute(text("""
                SELECT MIN(id::integer), MAX(id::integer) 
                FROM documents
                WHERE id ~ '^[0-9]+$'
            """)).fetchone()
            
            min_id, max_id = id_range if id_range and id_range[0] else (None, None)
            
            print(f"Index status: {doc_count} documents")
            if null_embeddings:
                print(f"WARNING: Found {len(null_embeddings)} documents with NULL embeddings!")
            
            return {
                "status": "healthy" if not null_embeddings else "inconsistent",
                "documents_count": doc_count,
                # Remove or update the cross-table consistency checks
                "consistency": {
                    "status": "ok" if not null_embeddings else "issues",
                    "null_embeddings_count": len(null_embeddings),
                    "null_embeddings_sample": null_embeddings
                },
                "sample_ids": sample_ids,
                "id_range": {
                    "min": min_id,
                    "max": max_id,
                    "range": (max_id - min_id + 1) if min_id is not None else None
                }
            }
    except Exception as e:
        print(f"Error checking index status: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    
    @app.post("/delete-all", status_code=status.HTTP_200_OK)
    def delete_all_data():
        """
        Debug/testing endpoint: Erase all indexed data from txtai and database.
        WARNING: This is destructive. Remove before production!
        """
        try:
            # Delete all data from txtai (semantic index)
            embeddings.delete("*")

            # Delete all rows from database tables
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM documents"))
                conn.execute(text("DELETE FROM embeddings"))
                conn.execute(text("DELETE FROM sections"))
                conn.commit()

            return {"status": "success", "message": "All data erased from txtai and database"}
        except Exception as e:
            print(f"Error deleting all data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to erase all data: {str(e)}")
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

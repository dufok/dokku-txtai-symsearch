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

if not DATABASE_URL:
    raise Exception("DATABASE_URL is not set")
else:
    # Convert postgres:// to postgresql:// for SQLAlchemy compatibility
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)


# Set up SQLAlchemy engine for direct PostgreSQL queries.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Check and initialize pgvector and pg_trgm extension on startup
print("Checking PostgreSQL pgvector and pg_trgm extension...")
try:
    with engine.connect() as conn:
        # Check for vector extension
        vector_result = conn.execute(text(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        )).fetchone()
        
        if vector_result:
            print(f"pgvector extension found with version {vector_result[1]}")
        else:
            print("pgvector extension not found, attempting to create it...")
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                print("Successfully created pgvector extension")
            except Exception as ext_error:
                print(f"WARNING: Could not create pgvector extension: {str(ext_error)}")
                print("Vector operations may not work correctly!")
        
        # Check for pg_trgm extension
        trgm_result = conn.execute(text(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'pg_trgm'"
        )).fetchone()
        
        if trgm_result:
            print(f"pg_trgm extension found with version {trgm_result[1]}")
        else:
            print("pg_trgm extension not found, attempting to create it...")
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                conn.commit()
                print("Successfully created pg_trgm extension")
            except Exception as ext_error:
                print(f"WARNING: Could not create pg_trgm extension: {str(ext_error)}")
                print("Fuzzy text search operations may not work correctly!")
                
except Exception as e:
    print(f"Error checking PostgreSQL extensions: {str(e)}")
    print("Continuing startup, but vector and fuzzy search operations may fail")

# Check if database has required schema and initialize if needed
print("Checking if database has required tables...")
try:
    with engine.connect() as conn:
        # Check if essential tables exist for txtai core functionality
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('sections', 'embeddings')
        """)).fetchall()
        
        existing_tables = [row[0] for row in result]
        print(f"Found tables: {existing_tables}")
        
        # Create txtai required tables
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
        
        if 'embeddings' not in existing_tables:
            print("Creating embeddings table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    indexid SERIAL PRIMARY KEY,
                    id TEXT UNIQUE NOT NULL,
                    embedding VECTOR(384)
                )
            """))
        
        # Check for specialized search tables
        specialized_tables = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('bodies', 'titles', 'authors')
        """)).fetchall()
        
        specialized_existing = [row[0] for row in specialized_tables]
        print(f"Found specialized search tables: {specialized_existing}")
        
        # 1. Create bodies table (SEMANTIC ONLY)
        if 'bodies' not in specialized_existing:
            print("Creating bodies table (semantic search)...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS bodies (
                    id TEXT PRIMARY KEY,
                    body TEXT NOT NULL,
                    embedding VECTOR(384)
                )
            """))
            
            # Create index for vector similarity search
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS bodies_embedding_idx ON bodies 
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """))
        
        # 2. Create titles table (FULL TEXT + FUZZY)
        if 'titles' not in specialized_existing:
            print("Creating titles table (fuzzy full-text search)...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS titles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL
                )
            """))
            
            # Create trigram index for fuzzy search
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS titles_trgm_idx ON titles 
                USING gin (title gin_trgm_ops)
            """))
        
        # 3. Create authors table (HYBRID)
        if 'authors' not in specialized_existing:
            print("Creating authors table (hybrid search)...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS authors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    bio TEXT,
                    embedding VECTOR(384),
                    name_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', name)) STORED
                )
            """))
            
            # Create vector index for semantic search
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS authors_embedding_idx ON authors 
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """))
            
            # Create trigram index for fuzzy name search
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS authors_name_trgm_idx ON authors 
                USING gin (name gin_trgm_ops)
            """))
            
            # Create full-text search index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS authors_name_tsv_idx ON authors 
                USING gin (name_tsv)
            """))
        
        conn.commit()
        print("Database schema initialized successfully")

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

# Build base txtai configuration 
base_config = {
    "path": MODEL_PATH,
    "cache": MODEL_CACHE_DIR,
    "content": DATABASE_URL,
    "backend": "pgvector",
    "incremental": True
}

# 1. Body config (SEMANTIC ONLY)
body_config = {
    **base_config,
    "pgvector": {
        "url": DATABASE_URL,
        "table": "bodies",
        "content": "bodies",
        "column": "embedding",
        "content_column": "body"
    }
}

# 2. No txtai config for titles (pure SQL fuzzy search)

# 3. Author config (HYBRID)
author_config = {
    **base_config,
    "pgvector": {
        "url": DATABASE_URL,
        "table": "authors",
        "content": "authors",
        "column": "embedding",
        "content_column": "bio"
    },
    "scoring": {"method": "bm25", "terms": True}
}

# Initialize specialized search instances
try:
    # Body search - semantic only
    body_embeddings = Embeddings(body_config)
    print(f"Body search initialized with model {MODEL_PATH}")
    
    # Author search - hybrid
    author_embeddings = Embeddings(author_config)
    print(f"Author search initialized with model {MODEL_PATH}")
    
    # No embeddings needed for title search (using fuzzy text search)
    
except Exception as e:
    print(f"Error initializing specialized search: {str(e)}")
    raise

# Define Pydantic models for specialized search
class BodyDocument(BaseModel):
    id: str
    body: str

class TitleDocument(BaseModel):
    id: str
    title: str

class AuthorDocument(BaseModel):
    id: str
    name: str
    bio: Optional[str] = None

class SearchRequest(BaseModel):
    text: str
    limit: int = 10


@app.get("/info")
def info():
    """
    Returns the current configuration and service status with index statistics
    for specialized search tables (bodies, titles, authors).
    """
    try:
        # Get counts from each specialized table
        with engine.connect() as conn:
            bodies_count = conn.execute(text("SELECT COUNT(*) FROM bodies")).scalar() or 0
            titles_count = conn.execute(text("SELECT COUNT(*) FROM titles")).scalar() or 0
            authors_count = conn.execute(text("SELECT COUNT(*) FROM authors")).scalar() or 0
            
            # Get sample IDs from each table (for debugging/verification)
            bodies_sample = [row[0] for row in conn.execute(text("SELECT id FROM bodies ORDER BY id LIMIT 3"))]
            titles_sample = [row[0] for row in conn.execute(text("SELECT id FROM titles ORDER BY id LIMIT 3"))]
            authors_sample = [row[0] for row in conn.execute(text("SELECT id FROM authors ORDER BY id LIMIT 3"))]
        
        return {
            "status": "Txtai service running",
            "configs": {
                "body": {
                    "model": MODEL_PATH,
                    "search_type": "semantic only",
                    "table": "bodies"
                },
                "title": {
                    "search_type": "fuzzy match (trigram)",
                    "table": "titles"
                },
                "author": {
                    "model": MODEL_PATH,
                    "search_type": "hybrid (semantic + trigram)",
                    "table": "authors"
                }
            },
            "index_stats": {
                "bodies_count": bodies_count,
                "titles_count": titles_count,
                "authors_count": authors_count,
                "total_count": bodies_count + titles_count + authors_count,
                "samples": {
                    "bodies": bodies_sample,
                    "titles": titles_sample,
                    "authors": authors_sample
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error getting info: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    

@app.post("/index-body")
def index_body(doc: BodyDocument):
    """
    Index a single body content using semantic search.
    """
    try:
        # Generate embedding for body content
        embedding = body_embeddings.transform(doc.body)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # Insert into bodies table
        with engine.connect() as conn:
            stmt = text("""
                INSERT INTO bodies (id, body, embedding)
                VALUES (:id, :body, CAST(:embedding AS vector(384)))
                ON CONFLICT (id) DO UPDATE
                SET body = :body, embedding = CAST(:embedding AS vector(384))
            """)
            
            conn.execute(stmt, {
                "id": doc.id,
                "body": doc.body,
                "embedding": embedding
            })
            conn.commit()
        
        return {"message": f"Body content {doc.id} indexed"}
    except Exception as e:
        print(f"Error indexing body: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index-title")
def index_title(doc: TitleDocument):
    """
    Index a single title using fuzzy search.
    """
    try:
        # Insert title (no embedding needed)
        with engine.connect() as conn:
            stmt = text("""
                INSERT INTO titles (id, title)
                VALUES (:id, :title)
                ON CONFLICT (id) DO UPDATE
                SET title = :title
            """)
            
            conn.execute(stmt, {
                "id": doc.id,
                "title": doc.title
            })
            conn.commit()
        
        return {"message": f"Title {doc.id} indexed"}
    except Exception as e:
        print(f"Error indexing title: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index-author")
def index_author(doc: AuthorDocument):
    """
    Index a single author using hybrid search.
    """
    try:
        # Use bio if provided, otherwise use name
        text_for_embedding = doc.bio if doc.bio else doc.name
        
        # Generate embedding for author
        embedding = author_embeddings.transform(text_for_embedding)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # Insert into authors table
        with engine.connect() as conn:
            stmt = text("""
                INSERT INTO authors (id, name, bio, embedding)
                VALUES (:id, :name, :bio, CAST(:embedding AS vector(384)))
                ON CONFLICT (id) DO UPDATE
                SET name = :name, bio = :bio, embedding = CAST(:embedding AS vector(384))
            """)
            
            conn.execute(stmt, {
                "id": doc.id,
                "name": doc.name,
                "bio": doc.bio,
                "embedding": embedding
            })
            conn.commit()
        
        return {"message": f"Author {doc.id} indexed"}
    except Exception as e:
        print(f"Error indexing author: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-index-bodies")
def bulk_index_bodies(docs: list[BodyDocument]):
    """
    Bulk index body contents using semantic search.
    """
    try:
        print(f"Processing bulk index of {len(docs)} bodies")
        
        # Process in batches
        batch_size = 10
        success_count = 0
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            
            with engine.connect() as conn:
                for doc in batch:
                    # Generate embedding
                    embedding = body_embeddings.transform(doc.body)
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    # Insert into bodies table
                    stmt = text("""
                        INSERT INTO bodies (id, body, embedding)
                        VALUES (:id, :body, CAST(:embedding AS vector(384)))
                        ON CONFLICT (id) DO UPDATE
                        SET body = :body, embedding = CAST(:embedding AS vector(384))
                    """)
                    
                    conn.execute(stmt, {
                        "id": doc.id,
                        "body": doc.body,
                        "embedding": embedding
                    })
                conn.commit()
            
            success_count += len(batch)
            print(f"Indexed {success_count}/{len(docs)} bodies")
        
        return {"message": f"All {len(docs)} body contents indexed successfully"}
    except Exception as e:
        print(f"Error bulk indexing bodies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bulk-index-titles")
def bulk_index_titles(docs: list[TitleDocument]):
    """
    Bulk index titles using fuzzy search.
    """
    try:
        print(f"Processing bulk index of {len(docs)} titles")
        
        # Process in one batch (no embeddings needed)
        with engine.connect() as conn:
            for doc in docs:
                stmt = text("""
                    INSERT INTO titles (id, title)
                    VALUES (:id, :title)
                    ON CONFLICT (id) DO UPDATE
                    SET title = :title
                """)
                
                conn.execute(stmt, {
                    "id": doc.id,
                    "title": doc.title
                })
            conn.commit()
        
        return {"message": f"All {len(docs)} titles indexed successfully"}
    except Exception as e:
        print(f"Error bulk indexing titles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-index-authors")
def bulk_index_authors(docs: list[AuthorDocument]):
    """
    Bulk index authors using hybrid search.
    """
    try:
        print(f"Processing bulk index of {len(docs)} authors")
        
        # Process in batches
        batch_size = 10
        success_count = 0
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            
            with engine.connect() as conn:
                for doc in batch:
                    # Use bio if provided, otherwise use name
                    text_for_embedding = doc.bio if doc.bio else doc.name
                    
                    # Generate embedding
                    embedding = author_embeddings.transform(text_for_embedding)
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    # Insert into authors table
                    stmt = text("""
                        INSERT INTO authors (id, name, bio, embedding)
                        VALUES (:id, :name, :bio, CAST(:embedding AS vector(384)))
                        ON CONFLICT (id) DO UPDATE
                        SET name = :name, bio = :bio, embedding = CAST(:embedding AS vector(384))
                    """)
                    
                    conn.execute(stmt, {
                        "id": doc.id,
                        "name": doc.name,
                        "bio": doc.bio,
                        "embedding": embedding
                    })
                conn.commit()
            
            success_count += len(batch)
            print(f"Indexed {success_count}/{len(docs)} authors")
        
        return {"message": f"All {len(docs)} authors indexed successfully"}
    except Exception as e:
        print(f"Error bulk indexing authors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-combined")
def search_combined(req: SearchRequest):
    """
    Combined search that prioritizes results by match quality:
    1. Exact title matches (highest priority)
    2. Titles containing all search terms (medium priority)
    3. Body semantic matches (lowest priority)
    
    Results are scored and ranked based on match quality.
    """
    try:
        # Parse search query into keywords for title matching
        query_text = req.text.strip().lower()
        search_terms = [term.lower() for term in query_text.split() if len(term) > 2]
        
        # Generate embedding for body search
        query_embedding = body_embeddings.transform(req.text)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        combined_results = {}
        
        with engine.connect() as conn:
            # Step 1: Execute title search for exact matches
            exact_match_stmt = text("""
                SELECT id, title, similarity(title, :query) AS base_score
                FROM titles
                WHERE LOWER(title) = :query_lower
                ORDER BY similarity(title, :query) DESC
                LIMIT 100
            """)
            
            exact_matches = conn.execute(exact_match_stmt, {
                "query": req.text,
                "query_lower": query_text
            })
            
            # Process exact title matches (highest priority)
            for row in exact_matches:
                combined_results[row.id] = {
                    "id": row.id,
                    "title": row.title,
                    "score": float(row.base_score) * 2.5,  # Highest boost for exact matches
                    "match_type": "exact_title"
                }
            
            # Step 2: Find titles containing all search terms
            if search_terms:
                # Build a query that checks if all terms are present
                conditions = []
                params = {"limit": 100}
                
                for i, term in enumerate(search_terms):
                    conditions.append(f"LOWER(title) LIKE '%' || :term{i} || '%'")
                    params[f"term{i}"] = term
                    
                where_clause = " AND ".join(conditions)
                
                all_terms_stmt = text(f"""
                    SELECT id, title, similarity(title, :query) AS base_score
                    FROM titles
                    WHERE {where_clause}
                    AND LOWER(title) != :query_lower
                    ORDER BY similarity(title, :query) DESC
                    LIMIT :limit
                """)
                
                params["query"] = req.text
                params["query_lower"] = query_text
                
                all_terms_matches = conn.execute(all_terms_stmt, params)
                
                # Process titles with all search terms
                for row in all_terms_matches:
                    if row.id not in combined_results:
                        combined_results[row.id] = {
                            "id": row.id,
                            "title": row.title,
                            "score": float(row.base_score) * 1.8,  # Medium boost
                            "match_type": "all_terms_title"
                        }
            
            # Step 3: Execute body search (semantic)
            body_stmt = text("""
                SELECT id, body, (1 - (embedding <#> CAST(:query_embedding AS vector(384)))) AS score
                FROM bodies
                ORDER BY embedding <#> CAST(:query_embedding AS vector(384)) ASC
                LIMIT :limit
            """)
            
            body_matches = conn.execute(body_stmt, {
                "query_embedding": query_embedding,
                "limit": req.limit * 2  # Get more results to ensure good coverage
            })
            
            # Process body results
            for row in body_matches:
                shout_id = row.id
                body_score = float(row.score) * 0.9  # Apply slight reduction to body scores
                
                if shout_id in combined_results:
                    # Already found in title search, keep higher score
                    if body_score > combined_results[shout_id]["score"]:
                        combined_results[shout_id]["score"] = body_score
                        combined_results[shout_id]["body"] = row.body
                        combined_results[shout_id]["match_type"] = "body+" + combined_results[shout_id]["match_type"]
                else:
                    # Body-only match
                    combined_results[shout_id] = {
                        "id": shout_id,
                        "body": row.body,
                        "score": body_score,
                        "match_type": "body"
                    }
        
        # Convert to list, sort by score, and apply pagination
        results = list(combined_results.values())
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:req.limit]
        
        return {"results": results}
    except Exception as e:
        print(f"Error in combined search: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-author")
def search_author(req: SearchRequest):
    """
    Search authors using hybrid approach (semantic + fuzzy).
    """
    try:
        # Generate query embedding for semantic search
        query_embedding = author_embeddings.transform(req.text)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Define weights for hybrid search
        fuzzy_weight = 0.4
        sem_weight = 0.6
        
        # Execute hybrid search
        with engine.connect() as conn:
            stmt = text("""
                WITH 
                    fuzzy AS (
                        SELECT id, name, similarity(name, :query) AS fuzzy_score
                        FROM authors
                        WHERE similarity(name, :query) > 0.3
                        ORDER BY fuzzy_score DESC
                        LIMIT 30
                    ),
                    semantic AS (
                        SELECT id, name, (1 - (embedding <#> CAST(:query_embedding AS vector(384)))) AS sem_score
                        FROM authors
                        ORDER BY embedding <#> CAST(:query_embedding AS vector(384)) ASC
                        LIMIT 30
                    )
                SELECT COALESCE(f.id, s.id) as id, 
                       COALESCE(f.name, s.name) as name,
                       COALESCE(f.fuzzy_score, 0) * :fuzzy_weight + 
                       COALESCE(s.sem_score, 0) * :sem_weight AS combined_score
                FROM fuzzy f
                FULL OUTER JOIN semantic s ON f.id = s.id
                ORDER BY combined_score DESC
                LIMIT :limit
            """)
            
            results = conn.execute(stmt, {
                "query": req.text,
                "query_embedding": query_embedding,
                "fuzzy_weight": fuzzy_weight,
                "sem_weight": sem_weight,
                "limit": req.limit
            })
            
            # Process results
            rows = [{
                "id": row.id,
                "name": row.name,
                "score": float(row.combined_score)
            } for row in results]
        
        return {"results": rows}
    except Exception as e:
        print(f"Error searching authors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dbtest")
def db_test():
    """Test database connection and specialized tables."""
    try:
        print(f"Testing database connection with URL: {DATABASE_URL}")
        results = {}
        
        with engine.connect() as conn:
            # Basic connection test
            result = conn.execute(text("SELECT 1")).fetchone()
            results["connection"] = "success"
            
            # Check if specialized tables exist
            tables_query = text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('bodies', 'titles', 'authors')
            """)
            existing_tables = [row[0] for row in conn.execute(tables_query)]
            results["existing_tables"] = existing_tables
            
            # Count rows in each specialized table
            if "bodies" in existing_tables:
                results["bodies_count"] = conn.execute(text("SELECT COUNT(*) FROM bodies")).scalar()
            
            if "titles" in existing_tables:
                results["titles_count"] = conn.execute(text("SELECT COUNT(*) FROM titles")).scalar()
                
            if "authors" in existing_tables:
                results["authors_count"] = conn.execute(text("SELECT COUNT(*) FROM authors")).scalar()
                
            return {
                "status": "success", 
                "url": DATABASE_URL.replace(DATABASE_URL.split('@')[0], "***"),
                "specialized_tables": results
            }
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health_check():
    """Check if the service is healthy with working database and models."""
    try:
        # Check database
        with engine.connect() as conn:
            db_ok = conn.execute(text("SELECT 1")).fetchone() is not None
        
        # Check models by testing embedding operations
        models_ok = False
        try:
            # Test both embedding models
            body_vector = body_embeddings.transform("test")
            author_vector = author_embeddings.transform("test")
            models_ok = (body_vector is not None and len(body_vector) > 0 and
                        author_vector is not None and len(author_vector) > 0)
        except Exception as model_error:
            print(f"Model health check failed: {str(model_error)}")
            models_ok = False
        
        # Check if specialized tables exist
        tables_exist = False
        with engine.connect() as conn:
            tables = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('bodies', 'titles', 'authors')
            """)).fetchall()
            tables_exist = len(tables) == 3  # All three specialized tables must exist
            
        return {
            "status": "healthy" if (db_ok and models_ok and tables_exist) else "unhealthy",
            "database": "connected" if db_ok else "disconnected",
            "models": "loaded" if models_ok else "not_loaded",
            "specialized_tables": "complete" if tables_exist else "incomplete"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
    
@app.post("/reset-connection")
def reset_connection():
    """Reset all database connections and embeddings objects."""
    global body_embeddings, author_embeddings, engine
    
    try:
        # Dispose of all connections in the pool
        engine.dispose()
        
        # Recreate the engine
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
        # Recreate embeddings objects with fresh connections
        body_embeddings = Embeddings(body_config)
        author_embeddings = Embeddings(author_config)
        
        return {
            "status": "success",
            "message": "Database connections reset and embeddings objects recreated"
        }
    except Exception as e:
        print(f"Connection reset error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    
@app.post("/verify-docs")
def verify_documents(req: VerifyRequest):
    """
    Verify which documents from the provided list exist in each specialized table.
    Returns lists of missing and existing document IDs per content type.
    """
    try:
        results = {}
        
        with engine.connect() as conn:
            # Check existence in bodies table
            placeholders = ", ".join([f"'{id}'" for id in req.doc_ids])
            
            # Bodies
            body_query = f"SELECT id FROM bodies WHERE id IN ({placeholders})"
            body_result = conn.execute(text(body_query))
            body_existing = [row[0] for row in body_result]
            body_missing = [id for id in req.doc_ids if id not in body_existing]
            
            # Titles
            title_query = f"SELECT id FROM titles WHERE id IN ({placeholders})"
            title_result = conn.execute(text(title_query))
            title_existing = [row[0] for row in title_result]
            title_missing = [id for id in req.doc_ids if id not in title_existing]
            
            # Authors (only if IDs might be author IDs)
            author_query = f"SELECT id FROM authors WHERE id IN ({placeholders})"
            author_result = conn.execute(text(author_query))
            author_existing = [row[0] for row in author_result]
            
            return {
                "total_requested": len(req.doc_ids),
                "bodies": {
                    "existing": body_existing,
                    "missing": body_missing,
                    "exists_count": len(body_existing),
                    "missing_count": len(body_missing)
                },
                "titles": {
                    "existing": title_existing,
                    "missing": title_missing,
                    "exists_count": len(title_existing),
                    "missing_count": len(title_missing)
                },
                "authors": {
                    "existing": author_existing,
                    "exists_count": len(author_existing)
                }
            }
    except Exception as e:
        print(f"Error verifying documents: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    
@app.get("/index-status")
def index_status():
    """Return detailed statistics about the current index state for all specialized tables."""
    try:
        print("Index status endpoint called - checking statistics")
        stats = {}
        
        with engine.connect() as conn:
            # Get counts from each table
            bodies_count = conn.execute(text("SELECT COUNT(*) FROM bodies")).scalar() or 0
            titles_count = conn.execute(text("SELECT COUNT(*) FROM titles")).scalar() or 0
            authors_count = conn.execute(text("SELECT COUNT(*) FROM authors")).scalar() or 0
            
            # Check for null embeddings in bodies table
            null_bodies = conn.execute(text("""
                SELECT id FROM bodies 
                WHERE embedding IS NULL
                LIMIT 10
            """)).fetchall()
            null_bodies_ids = [row[0] for row in null_bodies]
            
            # Check for null embeddings in authors table
            null_authors = conn.execute(text("""
                SELECT id FROM authors 
                WHERE embedding IS NULL
                LIMIT 10
            """)).fetchall()
            null_authors_ids = [row[0] for row in null_authors]
            
            # Get sample IDs from each table
            bodies_sample = [row[0] for row in conn.execute(text(
                "SELECT id FROM bodies ORDER BY id LIMIT 5"
            ))]
            titles_sample = [row[0] for row in conn.execute(text(
                "SELECT id FROM titles ORDER BY id LIMIT 5"
            ))]
            authors_sample = [row[0] for row in conn.execute(text(
                "SELECT id FROM authors ORDER BY id LIMIT 5"
            ))]
            
        return {
            "status": "healthy" if not (null_bodies_ids or null_authors_ids) else "inconsistent",
            "counts": {
                "bodies": bodies_count,
                "titles": titles_count,
                "authors": authors_count,
                "total": bodies_count + titles_count + authors_count
            },
            "consistency": {
                "status": "ok" if not (null_bodies_ids or null_authors_ids) else "issues",
                "null_embeddings": {
                    "bodies": null_bodies_ids,
                    "authors": null_authors_ids
                }
            },
            "samples": {
                "bodies": bodies_sample,
                "titles": titles_sample,
                "authors": authors_sample
            }
        }
    except Exception as e:
        print(f"Error checking index status: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    
@app.post("/delete-all")
def delete_all_data():
    """
    Debug/testing endpoint: Erase all indexed data from specialized tables.
    WARNING: This is destructive. Remove before production!
    """
    try:
        deletion_stats = {}
        
        # Delete data from txtai embeddings instances
        try:
            body_embeddings.delete("*")
            deletion_stats["body_embeddings_deleted"] = True
        except Exception as e:
            deletion_stats["body_embeddings_deleted"] = False
            deletion_stats["body_embeddings_error"] = str(e)
            
        try:
            author_embeddings.delete("*")
            deletion_stats["author_embeddings_deleted"] = True
        except Exception as e:
            deletion_stats["author_embeddings_deleted"] = False
            deletion_stats["author_embeddings_error"] = str(e)
        
        # Delete all rows from database tables
        with engine.connect() as conn:
            # Delete from specialized tables
            bodies_deleted = conn.execute(text("DELETE FROM bodies")).rowcount
            titles_deleted = conn.execute(text("DELETE FROM titles")).rowcount
            authors_deleted = conn.execute(text("DELETE FROM authors")).rowcount
            
            # Delete from txtai internal tables
            conn.execute(text("DELETE FROM embeddings"))
            conn.execute(text("DELETE FROM sections"))
            conn.commit()
            
            deletion_stats.update({
                "bodies_rows_deleted": bodies_deleted,
                "titles_rows_deleted": titles_deleted,
                "authors_rows_deleted": authors_deleted
            })

        return {
            "status": "success", 
            "message": "All data erased from specialized search tables",
            "details": deletion_stats
        }
    except Exception as e:
        print(f"Error deleting all data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to erase all data: {str(e)}")
    

        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

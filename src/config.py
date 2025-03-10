import os
from urllib.parse import urlparse

class Config:
    # Model configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/var/lib/model")
    
    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Parse Redis URL if provided
    if REDIS_URL:
        parsed_url = urlparse(REDIS_URL)
        REDIS_HOST = parsed_url.hostname
        REDIS_PORT = parsed_url.port or 6379
        REDIS_PASSWORD = parsed_url.password
        REDIS_DB = int(parsed_url.path.lstrip('/') or 0)
    else:
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
        REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    # Service configuration
    SEARCH_ENABLED = os.getenv("SEARCH_ENABLED", "true").lower() in ["true", "1", "yes"]
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", 100))
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
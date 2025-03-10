import logging
import os
from typing import Dict, List, Optional, Any

import redis
from txtai.embeddings import Embeddings

from config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("txtai-service")


class TxtAIService:
    def __init__(self):
        self.embeddings = None
        self.available = Config.SEARCH_ENABLED
        self.initialized = False
        self.index_size = 0
        
        # Setup Redis
        self.redis = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            password=Config.REDIS_PASSWORD,
            db=Config.REDIS_DB,
            decode_responses=True
        )
        
        # Create model cache directory
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        logger.info("TxtAI service initialized with config: SEARCH_ENABLED=%s, MODEL_PATH=%s", 
                   Config.SEARCH_ENABLED, Config.MODEL_PATH)
        
        # Initialize embeddings if search is enabled
        if self.available:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the txtai embeddings model"""
        try:
            logger.info("Loading embeddings model %s", Config.MODEL_PATH)
            self.embeddings = Embeddings(
                path=Config.MODEL_PATH,
                content=True,
                cache=Config.MODEL_CACHE_DIR
            )
            self.initialized = True
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error("Failed to initialize embeddings: %s", str(e))
            self.initialized = False
    
    def index_document(self, document_id: str, text: str) -> bool:
        """Index a single document"""
        if not self.available or not self.initialized:
            return False
            
        try:
            logger.debug("Indexing document %s", document_id)
            self.embeddings.index([(document_id, text, None)])
            self.index_size += 1
            return True
        except Exception as e:
            logger.error("Error indexing document %s: %s", document_id, str(e))
            return False
    
    def bulk_index(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Index multiple documents"""
        if not self.available or not self.initialized:
            return {"status": "error", "message": "Search service is not available"}
            
        try:
            data = [(doc["id"], doc["text"], None) for doc in documents]
            self.embeddings.index(data)
            self.index_size += len(documents)
            logger.info("Bulk indexed %d documents", len(documents))
            return {"status": "success", "count": len(documents)}
        except Exception as e:
            logger.error("Error during bulk indexing: %s", str(e))
            return {"status": "error", "message": str(e)}
    
    def search(self, query: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Search for documents matching the query"""
        if not self.available or not self.initialized:
            return []
            
        try:
            # txtai's search returns a list of (id, score) tuples
            # We need to adjust for pagination ourselves
            results = self.embeddings.search(query, limit + offset)
            
            # Apply offset and limit
            results = results[offset:offset+limit]
            
            # Format results
            formatted_results = [
                {"id": result[0], "score": float(result[1])}
                for result in results
            ]
            
            return formatted_results
        except Exception as e:
            logger.error("Error searching: %s", str(e))
            return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the search service"""
        return {
            "status": "available" if (self.available and self.initialized) else "unavailable",
            "model": Config.MODEL_PATH,
            "initialized": self.initialized,
            "index_size": self.index_size
        }


# Create service singleton
txtai_service = TxtAIService()
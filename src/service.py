import logging
import os
from typing import Dict, List, Any

from txtai.embeddings import Embeddings
from config import get_config

# Setup logging (level can be controlled via environment variables if needed)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("txtai-service")


class TxtAIService:
    def __init__(self):
        self.embeddings = None
        self.available = os.getenv("SEARCH_ENABLED", "true").lower() in ["true", "1", "yes"]
        self.initialized = False
        self.index_size = 0

        # Get configuration from config.py
        config = get_config()
        model_cache_dir = config.get("cache", "/var/lib/model")
        os.makedirs(model_cache_dir, exist_ok=True)

        logger.info(
            "TxtAI service initialized with SEARCH_ENABLED=%s, MODEL_PATH=%s",
            self.available,
            config.get("path"),
        )

        # Initialize embeddings if search is enabled
        if self.available:
            self._initialize_embeddings(config)

    def _initialize_embeddings(self, config: Dict[str, Any]):
        """Initialize the txtai embeddings model using the provided configuration."""
        try:
            logger.info("Loading embeddings model %s", config.get("path"))
            self.embeddings = Embeddings(config)
            self.initialized = True
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error("Failed to initialize embeddings: %s", str(e))
            self.initialized = False

    def index_document(self, document_id: str, text: str) -> bool:
        """Index a single document using upsert."""
        if not self.available or not self.initialized:
            return False

        try:
            logger.debug("Indexing document %s", document_id)
            # Use upsert so that only new or updated data is processed
            self.embeddings.upsert([(document_id, text, None)])
            self.index_size += 1
            return True
        except Exception as e:
            logger.error("Error indexing document %s: %s", document_id, str(e))
            return False

    def bulk_index(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Index multiple documents at once."""
        if not self.available or not self.initialized:
            return {"status": "error", "message": "Search service is not available"}

        try:
            data = [(doc["id"], doc["text"], None) for doc in documents]
            self.embeddings.upsert(data)
            self.index_size += len(documents)
            logger.info("Bulk indexed %d documents", len(documents))
            return {"status": "success", "count": len(documents)}
        except Exception as e:
            logger.error("Error during bulk indexing: %s", str(e))
            return {"status": "error", "message": str(e)}

    def search(self, query: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Search for documents matching the query."""
        if not self.available or not self.initialized:
            return []

        try:
            # txtai's search returns a list of (id, score) tuples.
            # We apply pagination (offset and limit) here.
            results = self.embeddings.search(query, limit + offset)
            results = results[offset:offset + limit]
            formatted_results = [
                {"id": result[0], "score": float(result[1])} for result in results
            ]
            return formatted_results
        except Exception as e:
            logger.error("Error searching: %s", str(e))
            return []

    def get_info(self) -> Dict[str, Any]:
        """Get information about the search service."""
        return {
            "status": "available" if (self.available and self.initialized) else "unavailable",
            "model": os.getenv("MODEL_PATH"),
            "initialized": self.initialized,
            "index_size": self.index_size,
        }


# Create service singleton
txtai_service = TxtAIService()

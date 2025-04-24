import logging
import os
from typing import Dict, List, Any

from txtai.embeddings import Embeddings
from config import get_config, get_body_config, get_author_config

# Setup logging (level can be controlled via environment variables if needed)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("txtai-service")


class TxtAIService:
    def __init__(self):
        self.body_embeddings = None
        self.author_embeddings = None
        self.available = os.getenv("SEARCH_ENABLED", "true").lower() in [
            "true",
            "1",
            "yes",
        ]
        self.initialized = {"body": False, "author": False}
        self.index_size = {"bodies": 0, "titles": 0, "authors": 0}

        # Get configurations from config.py
        self.body_config = get_body_config()
        self.author_config = get_author_config()

        # For backward compatibility
        self.config = get_config()

        # Create model cache directory
        model_cache_dir = self.config.get("cache", "/var/lib/model")
        os.makedirs(model_cache_dir, exist_ok=True)

        logger.info(
            "TxtAI service initialized with SEARCH_ENABLED=%s, MODEL_PATH=%s",
            self.available,
            self.config.get("path"),
        )

        # Initialize embeddings if search is enabled
        if self.available:
            self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize the txtai embeddings models using the provided configurations."""
        # Initialize body embeddings (semantic search)
        try:
            logger.info(
                "Loading body embeddings model %s", self.body_config.get("path")
            )
            self.body_embeddings = Embeddings(self.body_config)
            self.initialized["body"] = True
            logger.info("Body embeddings model loaded successfully")
        except Exception as e:
            logger.error("Failed to initialize body embeddings: %s", str(e))

        # Initialize author embeddings (hybrid search)
        try:
            logger.info(
                "Loading author embeddings model %s", self.author_config.get("path")
            )
            self.author_embeddings = Embeddings(self.author_config)
            self.initialized["author"] = True
            logger.info("Author embeddings model loaded successfully")
        except Exception as e:
            logger.error("Failed to initialize author embeddings: %s", str(e))

    # BODY CONTENT METHODS (SEMANTIC SEARCH)

    def index_body(self, document_id: str, text: str) -> bool:
        """Index a single body document using semantic search."""
        if not self.available or not self.initialized["body"]:
            return False

        try:
            logger.debug("Indexing body %s", document_id)
            # Use txtai's API for indexing
            self.body_embeddings.upsert([(document_id, text, None)])
            self.index_size["bodies"] += 1
            return True
        except Exception as e:
            logger.error("Error indexing body %s: %s", document_id, str(e))
            return False

    def bulk_index_bodies(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Index multiple body documents at once using semantic search."""
        if not self.available or not self.initialized["body"]:
            return {
                "status": "error",
                "message": "Body search service is not available",
            }

        try:
            data = [(doc["id"], doc["text"], None) for doc in documents]
            self.body_embeddings.upsert(data)
            self.index_size["bodies"] += len(documents)
            logger.info("Bulk indexed %d body documents", len(documents))
            return {"status": "success", "count": len(documents)}
        except Exception as e:
            logger.error("Error during bulk body indexing: %s", str(e))
            return {"status": "error", "message": str(e)}

    def search_body(
        self, query: str, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search body content using semantic search."""
        if not self.available or not self.initialized["body"]:
            return []

        try:
            # txtai's search returns a list of (id, score) tuples
            # We apply pagination (offset and limit) here
            results = self.body_embeddings.search(query, limit + offset)
            results = results[offset : offset + limit]
            formatted_results = [
                {"id": result[0], "score": float(result[1])} for result in results
            ]
            return formatted_results
        except Exception as e:
            logger.error("Error searching bodies: %s", str(e))
            return []

    # AUTHOR METHODS (HYBRID SEARCH)

    def index_author(self, author_id: str, name: str, bio: str = None) -> bool:
        """Index a single author using hybrid search."""
        if not self.available or not self.initialized["author"]:
            return False

        try:
            logger.debug("Indexing author %s", author_id)
            # Use bio if provided, otherwise use name
            text_for_embedding = bio if bio else name

            # Use txtai's API for indexing
            self.author_embeddings.upsert([(author_id, text_for_embedding, None)])
            self.index_size["authors"] += 1
            return True
        except Exception as e:
            logger.error("Error indexing author %s: %s", author_id, str(e))
            return False

    def bulk_index_authors(self, authors: List[Dict[str, str]]) -> Dict[str, Any]:
        """Index multiple authors at once using hybrid search."""
        if not self.available or not self.initialized["author"]:
            return {
                "status": "error",
                "message": "Author search service is not available",
            }

        try:
            # Process authors through txtai API
            data = []
            for author in authors:
                author_id = author["id"]
                bio = author.get("bio")
                name = author.get("name", author.get("text", ""))
                text_for_embedding = bio if bio else name
                data.append((author_id, text_for_embedding, None))

            self.author_embeddings.upsert(data)
            self.index_size["authors"] += len(authors)
            logger.info("Bulk indexed %d authors", len(authors))
            return {"status": "success", "count": len(authors)}
        except Exception as e:
            logger.error("Error during bulk author indexing: %s", str(e))
            return {"status": "error", "message": str(e)}

    def search_author(
        self, query: str, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search authors using hybrid approach (semantic + fuzzy)."""
        if not self.available or not self.initialized["author"]:
            return []

        try:
            # For hybrid search, txtai handles combining semantic and lexical scores
            results = self.author_embeddings.search(query, limit + offset)
            results = results[offset : offset + limit]
            formatted_results = [
                {"id": result[0], "score": float(result[1])} for result in results
            ]
            return formatted_results
        except Exception as e:
            logger.error("Error searching authors: %s", str(e))
            return []

    # BACKWARDS COMPATIBILITY METHODS

    def index_document(self, document_id: str, text: str) -> bool:
        """
        Index a single document using upsert.
        For backward compatibility - indexes as body content.
        """
        return self.index_body(document_id, text)

    def bulk_index(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Index multiple documents at once.
        For backward compatibility - indexes as body content.
        """
        return self.bulk_index_bodies(documents)

    def search(
        self, query: str, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        For backward compatibility - searches body content.
        """
        return self.search_body(query, limit, offset)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the search service."""
        return {
            "status": (
                "available"
                if (self.available and any(self.initialized.values()))
                else "unavailable"
            ),
            "model_path": self.config.get("path"),
            "initialized": self.initialized,
            "index_size": self.index_size,
        }

    def delete_all(self) -> Dict[str, Any]:
        """
        Deletes all documents and embeddings from the database.
        WARNING: This is destructive and should only be used for testing/debugging.
        """
        if not self.available:
            return {"status": "error", "message": "Search service is not available"}

        deletion_stats = {}

        try:
            # Delete body data if initialized
            if self.initialized["body"]:
                try:
                    self.body_embeddings.delete("*")  # txtai: "*" deletes all documents
                    deletion_stats["bodies_deleted"] = True
                    self.index_size["bodies"] = 0
                except Exception as e:
                    deletion_stats["bodies_deleted"] = False
                    deletion_stats["bodies_error"] = str(e)

            # Delete author data if initialized
            if self.initialized["author"]:
                try:
                    self.author_embeddings.delete(
                        "*"
                    )  # txtai: "*" deletes all documents
                    deletion_stats["authors_deleted"] = True
                    self.index_size["authors"] = 0
                except Exception as e:
                    deletion_stats["authors_deleted"] = False
                    deletion_stats["authors_error"] = str(e)

            logger.warning("All embeddings have been deleted.")
            return {"status": "success", "details": deletion_stats}
        except Exception as e:
            logger.error("Failed to delete all data: %s", str(e))
            return {"status": "error", "message": str(e)}


# Create service singleton
txtai_service = TxtAIService()

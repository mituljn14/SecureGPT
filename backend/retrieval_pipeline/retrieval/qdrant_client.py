from qdrant_client import QdrantClient

from retrieval_pipeline.retrieval.config import QDRANT_URL, QDRANT_TIMEOUT

# Keep a single client instance
_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(QDRANT_URL, timeout=QDRANT_TIMEOUT)
    return _client

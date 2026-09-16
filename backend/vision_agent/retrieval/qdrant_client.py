"""
Read-only Qdrant client wrapper for semantic search over existing vector stores.
SAFETY: Contains ONLY read operations (get_collections, search, scroll, count).
Never deletes, recreates, or modifies points or collections.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from vision_agent.config.settings import settings
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("qdrant_client")


class QdrantClientWrapper:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.url = (url or settings.QDRANT_URL).rstrip("/")
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.timeout = timeout or settings.QDRANT_TIMEOUT
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            kwargs: Dict[str, Any] = {
                "url": self.url,
                "timeout": self.timeout,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = QdrantClient(**kwargs)
        return self._client

    def check_connection(self) -> bool:
        """Check if Qdrant server is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.debug(f"Qdrant connection check failed: {exc}")
            return False

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections and their vector dimensions, metric, and point counts.
        """
        results: List[Dict[str, Any]] = []
        try:
            collections = self.client.get_collections().collections
            for col in collections:
                info = self.client.get_collection(col.name)
                vectors_cfg = info.config.params.vectors

                if hasattr(vectors_cfg, "size"):
                    dimension = vectors_cfg.size
                    distance = str(vectors_cfg.distance)
                elif isinstance(vectors_cfg, dict):
                    dimension = {k: v.size for k, v in vectors_cfg.items()}
                    distance = {k: str(v.distance) for k, v in vectors_cfg.items()}
                else:
                    dimension = getattr(vectors_cfg, "size", "unknown")
                    distance = getattr(vectors_cfg, "distance", "unknown")

                results.append(
                    {
                        "name": col.name,
                        "points_count": info.points_count,
                        "vectors_count": getattr(info, "vectors_count", None),
                        "dimension": dimension,
                        "distance": distance,
                        "status": str(info.status),
                    }
                )
        except Exception as exc:
            logger.error(f"Failed to list Qdrant collections: {exc}")
            raise
        return results

    def collection_exists(self, collection_name: str) -> bool:
        """Checks whether the specified collection exists."""
        try:
            return self.client.collection_exists(collection_name)
        except Exception:
            return False

    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on an existing collection.
        Returns a list of matching points with score and payload.
        """
        if not self.collection_exists(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' does not exist in Qdrant at {self.url}. "
                f"Use 'python main.py inspect' to list available collections."
            )

        query_filter = None
        if filter_dict:
            must_conditions = [
                qmodels.FieldCondition(
                    key=k,
                    match=qmodels.MatchValue(value=v),
                )
                for k, v in filter_dict.items()
            ]
            query_filter = qmodels.Filter(must=must_conditions)

        # Use modern query_points if available, otherwise search
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            hits = res.points
        else:
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

        results: List[Dict[str, Any]] = []
        for hit in hits:
            results.append(
                {
                    "id": str(hit.id),
                    "score": hit.score,
                    "payload": hit.payload or {},
                }
            )
        return results


_wrapper_singleton: Optional[QdrantClientWrapper] = None


def get_qdrant_wrapper() -> QdrantClientWrapper:
    global _wrapper_singleton
    if _wrapper_singleton is None:
        _wrapper_singleton = QdrantClientWrapper()
    return _wrapper_singleton
"""
Knowledge retriever -- delegates to the v2 retrieval pipeline.

The original implementation embedded queries with its own
EmbeddingGenerator and searched `securegpt_corpus` (the v1 collection) via
an unnamed-vector query, with `role_filter` an OPTIONAL argument that
defaulted to None.

Three things were wrong with that in this system:

  1. Wrong collection. The live corpus is `securegpt_v2`, which stores
     NAMED vectors (`dense` + `sparse`). An unnamed-vector query does not
     resolve against it.
  2. A second embedding path. Two code paths producing "1024-dim BGE-M3
     vectors" by different routes will happily insert and search without
     erroring, and silently return nonsense if the vector spaces differ.
     There is now exactly one embedder in this system.
  3. Optional RBAC. `role_filter=None` searched the entire corpus across
     every department. `role` is now REQUIRED.

Everything below delegates to retrieval_pipeline, so the vision agent
retrieves through the same hybrid dense+sparse, RBAC-filtered path as the
enterprise chat.
"""
from __future__ import annotations

from typing import List, Optional

from vision_agent.core.models import RetrievalChunk
from vision_agent.utils.logging_utils import get_logger

from retrieval_pipeline.retrieval import retrieve_with_confidence
from retrieval_pipeline.retrieval.retrieve import COLLECTION_NAME

logger = get_logger("retriever")


class KnowledgeRetriever:
    def __init__(self, collection_name: Optional[str] = None, **_legacy):
        # Kept for signature compatibility with the standalone version;
        # the collection is whatever retrieval_pipeline is pointed at.
        self.collection_name = collection_name or COLLECTION_NAME

    def retrieve(
        self,
        query: str,
        role: str,
        top_k: int = 4,
        department_filter: Optional[str] = None,
        **_legacy,
    ) -> List[RetrievalChunk]:
        """RBAC-filtered hybrid retrieval. `role` is required."""
        if not query.strip():
            return []
        if not role:
            raise ValueError(
                "KnowledgeRetriever.retrieve() requires a role -- there is no "
                "RBAC-free retrieval path."
            )

        try:
            chunks, confidence = retrieve_with_confidence(
                query=query,
                role=role,
                department=department_filter,
                top_k=top_k,
            )
        except Exception as exc:
            logger.error(f"Retrieval error: {exc}")
            return []

        out: List[RetrievalChunk] = []
        for c in chunks:
            out.append(
                RetrievalChunk(
                    id=c.chunk_id or "",
                    text=c.text or "",
                    score=c.score,
                    file_name=c.file_name or "",
                    department=c.department or "",
                    page=c.page,
                    # v2 renamed document_id -> doc_id; the dataclass field
                    # keeps its old name so callers are unaffected.
                    document_id=c.doc_id or "",
                    metadata=c.payload or {},
                )
            )
        logger.info(
            f"Retrieved {len(out)} chunk(s) from '{self.collection_name}' "
            f"(dense confidence {confidence:.4f}, role={role})."
        )
        return out

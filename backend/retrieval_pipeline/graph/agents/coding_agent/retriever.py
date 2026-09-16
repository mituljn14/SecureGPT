"""
retriever.py -- coding-agent retrieval, v2 native.

Given a user query + role, find the chunk(s) identifying WHICH source
document the agent should mount into its sandbox for full-file analysis.

This replaces the standalone version that shipped with the coding agent,
which had four defects against the v2 corpus:

  1. Queried `securegpt_corpus` (the v1 collection). The live corpus is
     `securegpt_v2`, which stores NAMED vectors (`dense` + `sparse`) --
     an unnamed-vector query against it does not even resolve.
  2. Embedded queries by POSTing to Ollama's /api/embeddings with
     `bge-m3`, while the v2 corpus was embedded with FlagEmbedding's
     BGEM3FlagModel. Same model name, different runtime: nothing
     guarantees the two put queries and documents in the same vector
     space, and the failure is silent -- you get confident nonsense, not
     an error. Now reuses `embed_query_hybrid`, the exact function the
     main retrieval path uses.
  3. Dense-only search, ignoring the sparse index the collection was
     built with. Now does real dense+sparse RRF fusion.
  4. NO RBAC -- any role could reach any department's documents through
     this path. `role` is now a REQUIRED argument and the access_roles
     filter is always applied. There is no RBAC-free retrieval path.

It also read payload fields `document_id` / `source_hash`, which do not
exist in the v2 payload (the real fields are `doc_id` / `content_hash`).
"""

import logging
from typing import Optional

from qdrant_client import models as qmodels

from retrieval_pipeline.retrieval.retrieve import COLLECTION_NAME
from retrieval_pipeline.retrieval.qdrant_client import get_client
from retrieval_pipeline.retrieval.embedding import embed_query_hybrid
from retrieval_pipeline.retrieval.filter_builder import build_filter
from retrieval_pipeline.retrieval.config import DENSE_RELEVANCE_THRESHOLD

log = logging.getLogger(__name__)


class Retriever:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name

    def search(
        self,
        query: str,
        role: str,
        top_k: int = 8,
        department: Optional[str] = None,
        metadata_filters: Optional[dict] = None,
    ) -> list:
        """Hybrid dense+sparse search, RBAC-filtered.

        `role` is required -- passing None would search the whole corpus.
        Returns dicts carrying the v2 payload fields the agent needs,
        including `source_path` so the file can be resolved without a
        registry lookup.
        """
        if not role:
            raise ValueError(
                "Retriever.search() requires a role -- there is no RBAC-free "
                "retrieval path."
            )

        dense, sparse_dict = embed_query_hybrid(query)
        sparse_indices = [int(k) for k in sparse_dict.keys()]
        sparse_values = [float(v) for v in sparse_dict.values()]

        q_filter = build_filter(
            role=role,
            department=department,
            metadata_filters=metadata_filters or {},
        )

        client = get_client()
        response = client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                qmodels.Prefetch(query=dense, using="dense",
                                 filter=q_filter, limit=top_k * 4),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(indices=sparse_indices,
                                               values=sparse_values),
                    using="sparse", filter=q_filter, limit=top_k * 4),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        out = []
        for point in response.points:
            p = point.payload or {}
            out.append({
                "chunk_id": str(point.id),
                "doc_id": p.get("doc_id"),
                # `document_id` kept as an alias so downstream code written
                # against the old payload shape keeps working.
                "document_id": p.get("doc_id"),
                "file_name": p.get("file_name"),
                "source_path": p.get("source_path"),
                "content_hash": p.get("content_hash"),
                "department": p.get("department"),
                "chunk_index": p.get("chunk_index"),
                "text": p.get("text"),
                "score": point.score,
            })
        return out

    def has_real_match(self, query: str, role: str,
                       department: Optional[str] = None) -> bool:
        """Raw dense cosine of the best match against the shared relevance
        floor. Uses dense cosine, NOT the RRF score above -- RRF is
        rank-based and looks confident for whatever ranks #1 even when
        nothing relevant exists.
        """
        dense, _ = embed_query_hybrid(query)
        probe = get_client().query_points(
            collection_name=self.collection_name,
            query=dense, using="dense",
            query_filter=build_filter(role=role, department=department,
                                      metadata_filters={}),
            limit=1,
        )
        return bool(probe.points) and probe.points[0].score >= DENSE_RELEVANCE_THRESHOLD

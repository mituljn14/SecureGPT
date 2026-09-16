"""
Lexical reranking using the BGE-M3 sparse weights already stored in Qdrant.

Replaces the in-memory BM25Okapi rerank that used to sit in
multi_agent_retrieval_node. BM25 was the wrong tool in that position for
three reasons:

  1. IDF over nothing. BM25Okapi derives its inverse-document-frequency
     statistics from the corpus it is handed -- there, the ~20-80 chunk
     candidate pool. IDF computed over 40 documents is noise; a term is
     "rare" or "common" purely by accident of which chunks the rewritten
     queries happened to surface. The stored lexical weights come from
     BGE-M3, which learned term importance over its full training corpus.

  2. Whitespace tokenisation. `text.lower().split(" ")` does not match how
     the corpus was tokenised at ingest, so "Q3," "(Q3)" and "Q3" were three
     different terms, and every number carrying a comma or unit failed to
     match. The sparse vectors are keyed by the model's own token ids.

  3. Two scoring functions for one job. The main retrieval path already
     ranks lexically via the `sparse` index; reranking with a second,
     unrelated scorer meant the pipeline disagreed with itself about what
     "lexically relevant" means.

Scoring is the same dot product Qdrant computes internally for sparse
vectors: the sum of (query weight x chunk weight) over shared token ids.
"""

from typing import List, Optional

from retrieval_pipeline.retrieval.retrieve import COLLECTION_NAME
from retrieval_pipeline.retrieval.qdrant_client import get_client
from retrieval_pipeline.retrieval.embedding import embed_query_hybrid
from retrieval_pipeline.retrieval.schemas import RetrievedChunk


def _fetch_sparse_vectors(chunk_ids: List[str]) -> dict:
    """chunk_id -> {token_id: weight} for the stored sparse vectors."""
    if not chunk_ids:
        return {}

    points = get_client().retrieve(
        collection_name=COLLECTION_NAME,
        ids=chunk_ids,
        with_vectors=["sparse"],
        with_payload=False,
    )

    out = {}
    for p in points:
        vec = p.vector or {}
        sparse = vec.get("sparse") if isinstance(vec, dict) else None
        if sparse is None:
            continue
        out[str(p.id)] = dict(zip(sparse.indices, sparse.values))
    return out


def sparse_score(query_weights: dict, chunk_weights: dict) -> float:
    """Dot product over shared token ids. Iterates the smaller dict."""
    if not query_weights or not chunk_weights:
        return 0.0
    if len(query_weights) > len(chunk_weights):
        query_weights, chunk_weights = chunk_weights, query_weights
    return sum(w * chunk_weights.get(tok, 0.0) for tok, w in query_weights.items())


def sparse_rerank(
    query: str,
    chunks: List[RetrievedChunk],
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """Rerank `chunks` by lexical similarity to `query`.

    Falls back to the incoming order if embedding or the vector fetch fails
    -- a degraded ranking is recoverable, losing the candidates is not.
    """
    if not chunks:
        return []

    try:
        _dense, query_sparse = embed_query_hybrid(query)
        query_weights = {int(k): float(v) for k, v in query_sparse.items()}
    except Exception as e:
        print(f"[sparse_rerank] query embedding failed, keeping input order: {e}")
        return list(chunks)[:top_k] if top_k else list(chunks)

    try:
        vectors = _fetch_sparse_vectors([c.chunk_id for c in chunks if c.chunk_id])
    except Exception as e:
        print(f"[sparse_rerank] sparse vector fetch failed, keeping input order: {e}")
        return list(chunks)[:top_k] if top_k else list(chunks)

    scored = [
        (sparse_score(query_weights, vectors.get(c.chunk_id or "", {})), i, c)
        for i, c in enumerate(chunks)
    ]
    # Tie-break on original position so equal scores stay deterministic.
    scored.sort(key=lambda t: (-t[0], t[1]))

    ranked = [c for _s, _i, c in scored]
    return ranked[:top_k] if top_k else ranked

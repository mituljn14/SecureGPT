from typing import List, Optional, Tuple

from qdrant_client import models
from retrieval_pipeline.retrieval.filter_builder import build_filter
from retrieval_pipeline.retrieval.utils import format_results, deduplicate_chunks
from retrieval_pipeline.retrieval.schemas import RetrievedChunk
from retrieval_pipeline.retrieval.config import TOP_K, QDRANT_COLLECTION
from retrieval_pipeline.retrieval.query_metadata_parser import parse_query_metadata

from retrieval_pipeline.retrieval.qdrant_client import get_client
from retrieval_pipeline.retrieval.embedding import embed_query_hybrid
from retrieval_pipeline.retrieval.identifier_lookup import lookup_identifiers

# Re-exported so existing imports keep working; the value lives in config.
COLLECTION_NAME = QDRANT_COLLECTION


def _retrieve_impl(
    query: str,
    role: str,
    department: Optional[str],
    top_k: int,
) -> Tuple[List[RetrievedChunk], float]:
    metadata = parse_query_metadata(query)
    client = get_client()

    try:
        dense, sparse_dict = embed_query_hybrid(query)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return [], 0.0

    sparse_indices = [int(k) for k in sparse_dict.keys()]
    sparse_values = [float(v) for v in sparse_dict.values()]

    q_filter = build_filter(
         role=role,
         department=department,
         metadata_filters=metadata,
    )

    # Confidence probe -- cheap (dense-only, limit=1, no extra embedding
    # call since `dense` is already computed above): the raw dense cosine
    # of the single best match. This is a genuine relevance signal, unlike
    # the fused RRF score below (rank-based, ~1/(rank+k) -- looks
    # confident for whatever ranks #1 even when nothing relevant exists).
    # See DENSE_RELEVANCE_THRESHOLD in config.py for the calibration.
    probe = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense, using="dense", query_filter=q_filter, limit=1,
    )
    
    # Fall back to the unfiltered search if the period filter eliminated
    # everything -- not every document carries a parseable fiscal_year in
    # its payload, and a query mentioning a year that happens to match no
    # indexed period should still return the best available evidence
    # rather than dead-ending. Precision when it can, recall when it must.
    has_period_filter = any(k in metadata for k in ("year", "year_range", "quarter"))
    if not probe.points and has_period_filter:
        print(f"[Retrieval] Period filter {metadata} matched nothing; falling back to unfiltered.")
        q_filter = build_filter(
             role=role,
             department=department,
             metadata_filters=None,
        )
        probe = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense, using="dense", query_filter=q_filter, limit=1,
        )

    confidence = probe.points[0].score if probe.points else 0.0

    # Fetch a wider pool than top_k so that near-duplicate collapsing
    # (below) still leaves top_k genuinely distinct results -- this corpus
    # has a measured boilerplate paragraph repeated 96x in one source PDF,
    # which without over-fetch + dedup could occupy the entire top_k.
    fetch_k = max(top_k * 4, 20)

    prefetch = [
        models.Prefetch(
            query=dense,
            using="dense",
            filter=q_filter,
            limit=fetch_k
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values
            ),
            using="sparse",
            filter=q_filter,
            limit=fetch_k
        )
    ]

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_k,
        with_payload=True,
    )

    chunks = format_results(response.points)

    # Exact-identifier hits go first. A query naming a record code
    # ("WO-202207-000550") is asking for that record, and neither dense
    # nor sparse similarity reliably surfaces it among thousands of
    # sibling records (see identifier_lookup.py for the eval row that
    # proved it). Same RBAC filter; ranked by dense cosine among the
    # matches. An exact hit is the strongest relevance signal this
    # pipeline has, so it also sets the confidence the fast-path gate
    # reads -- the cascade must not escalate past evidence it already
    # holds.
    exact = lookup_identifiers(query, role, department, dense=dense)
    if exact:
        exact_ids = {c.chunk_id for c in exact}
        chunks = exact + [c for c in chunks if c.chunk_id not in exact_ids]
        confidence = 1.0

    chunks = deduplicate_chunks(chunks)
    return chunks[:top_k], confidence


def retrieve(
    query: str,
    role: str,
    department: Optional[str] = None,
    top_k: int = TOP_K,
) -> List[RetrievedChunk]:
    chunks, _confidence = _retrieve_impl(query, role, department, top_k)
    return chunks


def retrieve_with_confidence(
    query: str,
    role: str,
    department: Optional[str] = None,
    top_k: int = TOP_K,
) -> Tuple[List[RetrievedChunk], float]:
    """Same as retrieve(), but also returns a real relevance confidence:
    the raw dense cosine similarity of the single best dense match (0..1).

    NOT the RRF fusion score used to rank the returned chunks -- RRF is
    rank-based (~1/(rank+k)) and looks confident for whatever ranks #1
    even when nothing relevant exists in the corpus for this query.

    Used by fast_retrieval_node (graph/nodes.py) to decide whether to
    trust the fast hybrid path or escalate to CRAG/multi-hop agentic
    retrieval -- see DENSE_RELEVANCE_THRESHOLD in config.py.
    """
    return _retrieve_impl(query, role, department, top_k)

from typing import List

from retrieval_pipeline.retrieval.schemas import RetrievedChunk

# Every RetrievedChunk field that comes straight off the payload with no
# renaming. chunk_id/score come from the Qdrant point itself, not payload.
_PAYLOAD_FIELDS = [
    "doc_id", "parent_block_id", "source_path", "file_name", "source_format",
    "doc_title", "content_hash", "department", "sub_department", "access_roles",
    "sensitivity", "functional_area", "region", "heading_path", "section_title",
    "chunk_type", "chunk_index", "total_chunks", "page", "sheet", "slide",
    "row_range", "row_total", "table_columns", "table_caption",
    "prev_chunk_id", "next_chunk_id", "fiscal_year", "fiscal_quarter",
    "fiscal_half", "doc_date", "token_count", "has_table", "identifiers",
    "context_header",
]


def format_results(results) -> List[RetrievedChunk]:
    formatted = []
    for hit in results:
        payload = hit.payload or {}
        kwargs = {k: payload.get(k) for k in _PAYLOAD_FIELDS}
        # dataclass defaults are empty list/False, not None -- don't clobber
        # them with an explicit None when the field is genuinely absent.
        for list_field in ("access_roles", "heading_path", "identifiers"):
            if kwargs.get(list_field) is None:
                kwargs[list_field] = []
        if kwargs.get("has_table") is None:
            kwargs["has_table"] = False

        formatted.append(RetrievedChunk(
            score=hit.score,
            text=payload.get("text", ""),
            chunk_id=str(hit.id),
            payload=payload,
            **kwargs,
        ))
    return formatted


def deduplicate_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Collapse chunks that are near-identical content surfaced from different
    documents/positions -- keeping the highest-ranked instance of each.

    This corpus has heavy source-level duplication (one boilerplate
    paragraph was measured occurring 96x inside a single source PDF, and
    the same report template repeats across regions), so without this an
    unlucky query can have its whole top-k occupied by copies of one
    sentence. `content_hash` (payload) was built specifically for this and
    was previously never read anywhere in the retrieval path.

    Chunks are assumed to already be sorted by relevance (as Qdrant
    returns them), so "first occurrence wins" keeps the best-ranked copy.
    """
    seen = set()
    out = []
    for c in chunks:
        h = c.content_hash
        if h and h in seen:
            continue
        if h:
            seen.add(h)
        out.append(c)
    return out

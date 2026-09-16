"""
Knowledge-base search for the Documents view.

Returns real RBAC-filtered hits -- score, source, department, page and the
chunk text -- so the document browser shows measured relevance instead of a
decorative percentage.
"""
import time

from fastapi import APIRouter, Depends, Query

from retrieval_pipeline.api.dependencies import get_current_user
from retrieval_pipeline.api.schemas import User

router = APIRouter(tags=["search"])


@router.get("/search")
async def search_corpus(
    q: str = Query(..., min_length=1),
    limit: int = Query(8, ge=1, le=50),
    current_user: User = Depends(get_current_user),
):
    from retrieval_pipeline.retrieval import retrieve_with_confidence

    started = time.perf_counter()
    try:
        chunks, confidence = retrieve_with_confidence(
            query=q, role=current_user.role, department=current_user.department, top_k=limit,
        )
    except Exception as exc:
        return {"query": q, "results": [], "took_ms": 0, "error": str(exc)}

    took_ms = int((time.perf_counter() - started) * 1000)

    results = []
    for c in chunks:
        results.append({
            "chunk_id": c.chunk_id,
            "file_name": c.file_name,
            "source_path": c.source_path,
            "department": c.department,
            "sub_department": c.sub_department,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "fiscal_year": c.fiscal_year,
            "fiscal_quarter": c.fiscal_quarter,
            "doc_date": c.doc_date,
            "sensitivity": c.sensitivity,
            # Fusion rank score -- shown as a rank, never as a confidence.
            "score": c.score,
            "text": c.text,
        })

    return {
        "query": q,
        "results": results,
        "took_ms": took_ms,
        # Raw dense cosine of the best match: the only number here that is a
        # genuine relevance signal (the per-result score is rank-based).
        "confidence": confidence,
        "role": current_user.role,
    }

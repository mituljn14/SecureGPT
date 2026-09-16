"""
Exact-identifier lookup -- the step the semantic cascade cannot do.

Every record-style chunk in securegpt_v2 carries an `identifiers` payload
list, extracted at ingest time (work orders "WO-202207-000550", POs
"PO-201707-1046", CVEs "CVE-2021-24203", project codes "PRJ-39835", ...),
and the field is keyword-indexed. A query that names one of those codes
is asking for *that record*, and dense embeddings are the wrong tool for
it: the RAGAS run that motivated this (evaluation/results/
eval_enterprise_1789287096.json, row 7) retrieved five semantically
similar work-order chunks and never the one with the requested ID --
context recall 0.0 -- while the same role asking about "Q3 2022 work
orders" in general DID get that chunk back. BGE-M3's sparse vector does
not rescue it either: the tokenizer splits the code into fragments that
thousands of sibling records share.

So: pull code-shaped tokens out of the query, ask Qdrant whether the
corpus has chunks tagged with exactly those identifiers, and if so put
them at the front of the context. The corpus decides what counts as an
identifier -- nothing here enumerates ID formats. The only assumption
is what all 60k+ stored identifiers share: a token made of letters,
digits and hyphens that contains at least one digit.

RBAC is applied exactly as in the semantic path: the same
build_filter(role) is AND-ed with the identifier condition, so a CHRO
asking for a COO work order by its exact ID still gets nothing.
"""
import os
import re
from typing import List, Optional

from qdrant_client.http import models as qmodels

from retrieval_pipeline.retrieval.config import QDRANT_COLLECTION
from retrieval_pipeline.retrieval.filter_builder import build_filter
from retrieval_pipeline.retrieval.qdrant_client import get_client
from retrieval_pipeline.retrieval.schemas import RetrievedChunk
from retrieval_pipeline.retrieval.utils import format_results

# How many exact-match chunks may be reserved at the front of the context.
# An identifier referenced from many documents (a project code cited in
# every status report) must not crowd out the semantic results entirely.
IDENTIFIER_LOOKUP_LIMIT = int(os.getenv("IDENTIFIER_LOOKUP_LIMIT", "3"))

# Shortest token worth looking up. Below this, "Q3", "A-1" and the like
# are period/section references, not record identifiers.
IDENTIFIER_MIN_LEN = int(os.getenv("IDENTIFIER_MIN_LEN", "5"))

# Code-shaped token: letters/digits joined by hyphens, starting with a
# letter or digit. Written without word-boundary escapes on purpose --
# the surrounding lookarounds do the same job and survive any shell or
# editor re-escaping (a previous regex in this codebase was silently
# corrupted that way).
_TOKEN = re.compile(r"(?<![A-Za-z0-9-])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+(?![A-Za-z0-9-])")
_HAS_DIGIT = re.compile(r"[0-9]")


class _ScoredRecord:
    """Minimal stand-in for a ScoredPoint when the match came from a
    filtered scroll rather than a vector query."""
    __slots__ = ("id", "payload", "score")

    def __init__(self, id, payload):
        self.id, self.payload, self.score = id, payload, 1.0


def extract_identifiers(query: str) -> List[str]:
    """Candidate identifiers in the query, upper-cased and de-duplicated.

    Upper-casing matches how the ingest pipeline stores them; a user
    typing "wo-202207-000550" means the same record.
    """
    out: List[str] = []
    for tok in _TOKEN.findall(query or ""):
        if len(tok) < IDENTIFIER_MIN_LEN or not _HAS_DIGIT.search(tok):
            continue
        up = tok.upper()
        if up not in out:
            out.append(up)
    return out


def _identifier_filter(role: str, department: Optional[str], candidates: List[str]) -> qmodels.Filter:
    base = build_filter(role=role, department=department, metadata_filters={})
    return qmodels.Filter(
        must=list(base.must or []) + [
            qmodels.FieldCondition(key="identifiers", match=qmodels.MatchAny(any=candidates)),
        ]
    )


def has_identifier_match(query: str, role: str, department: Optional[str] = None) -> bool:
    """Cheap existence check -- no embedding call, one filtered scroll.

    Used by the supervisor's safety net so a query naming a record the
    caller is cleared to read is never answered "directly" (ungrounded).
    """
    candidates = extract_identifiers(query)
    if not candidates:
        return False
    try:
        points, _ = get_client().scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=_identifier_filter(role, department, candidates),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
    except Exception as e:
        print(f"[Identifier lookup] existence check failed, skipping: {e}")
        return False
    return bool(points)


def lookup_identifiers(
    query: str,
    role: str,
    department: Optional[str],
    dense: Optional[List[float]] = None,
    limit: int = IDENTIFIER_LOOKUP_LIMIT,
) -> List[RetrievedChunk]:
    """Chunks tagged with an identifier named in the query, RBAC-filtered.

    With `dense` (the query embedding the caller already computed) the
    matches are ranked by dense cosine, so when one code is referenced
    from many chunks the most relevant ones come first and carry a real
    score. Without it, the first `limit` matches are returned unranked.
    """
    candidates = extract_identifiers(query)
    if not candidates:
        return []

    client = get_client()
    q_filter = _identifier_filter(role, department, candidates)
    try:
        if dense is not None:
            resp = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=dense, using="dense",
                query_filter=q_filter, limit=limit, with_payload=True,
            )
            points = resp.points
        else:
            records, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=q_filter, limit=limit,
                with_payload=True, with_vectors=False,
            )
            # scroll() returns Records without a score; format_results()
            # reads hit.score, so wrap each exact match with a full-
            # confidence one (Record is a pydantic model, can't be patched).
            points = [_ScoredRecord(r.id, r.payload) for r in records]
    except Exception as e:
        print(f"[Identifier lookup] failed for {candidates}: {e}")
        return []

    chunks = format_results(points)
    if chunks:
        found = sorted({i for c in chunks for i in (c.identifiers or []) if i in candidates})
        print(f"[Identifier lookup] {len(chunks)} exact chunk(s) for {found or candidates}")
    return chunks

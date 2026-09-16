"""
Build Qdrant metadata filters.

Input:
{
    "report_name": "...",
    "quarter": "...",
    "year": "...",
    "month": "...",
    "department": "..."
}

Output:
qmodels.Filter(...)
"""

from typing import Dict, Optional

from qdrant_client.http import models as qmodels

from retrieval_pipeline.retrieval.config import RBAC_SCOPE


def build_filter(
    role: str,
    department: Optional[str] = None,
    metadata_filters: Optional[Dict] = None,
):
    """
    Creates a Qdrant Filter.

    Parameters
    ----------
    role : str
        User role for RBAC.

    department : Optional[str]
        Department restriction.

    metadata_filters : Optional[dict]
        Metadata extracted from query (query_metadata_parser.parse_query_metadata).

    Returns
    -------
    qmodels.Filter
    """

    must = []

    # --------------------------------------------------
    # RBAC -- always applied, this is the security boundary
    # --------------------------------------------------

    role_clause = qmodels.FieldCondition(
        key="access_roles",
        match=qmodels.MatchValue(value=role),
    )
    if RBAC_SCOPE == "department" and department:
        # Department-level sharing (see RBAC_SCOPE in config.py): the
        # user's own role tag OR any chunk belonging to their department.
        # Nested Filter(should=[...]) inside must[] is how Qdrant expresses
        # (A or B) and C.
        must.append(qmodels.Filter(should=[
            role_clause,
            qmodels.FieldCondition(key="department", match=qmodels.MatchValue(value=department)),
        ]))
    else:
        must.append(role_clause)

    # --------------------------------------------------
    # Fiscal year / quarter -- applied ONLY when the query parser
    # explicitly detected one, using the payload's INTEGER fiscal_year /
    # fiscal_quarter fields (parsed at ingest time from the document's own
    # dates, not string-matched against a filename). These are unambiguous
    # once detected -- "Q1 2024" in the query can only mean fiscal_year=2024,
    # fiscal_quarter=1 -- so filtering on them is a precision win with no
    # real false-negative risk, unlike free-text department matching.
    # --------------------------------------------------

    metadata_filters = metadata_filters or {}

    year_range = metadata_filters.get("year_range")
    if year_range:
        lo, hi = year_range
        must.append(
            qmodels.FieldCondition(
                key="fiscal_year",
                range=qmodels.Range(gte=int(lo), lte=int(hi)),
            )
        )
    else:
        year = metadata_filters.get("year")
        if year:
            must.append(
                qmodels.FieldCondition(
                    key="fiscal_year",
                    match=qmodels.MatchValue(value=int(year)),
                )
            )

    quarter = metadata_filters.get("quarter")
    if quarter:
        must.append(
            qmodels.FieldCondition(
                key="fiscal_quarter",
                match=qmodels.MatchValue(value=int(quarter)),
            )
        )

    # --------------------------------------------------
    # Department is never applied as a RESTRICTION:
    #
    # 1. `department` is the calling user's home department (from their
    #    JWT), not a query-scope override. Under RBAC_SCOPE="department"
    #    it WIDENS access (OR-ed with the role tag above) so colleagues
    #    share a department's documents; it never narrows a CEO, whose
    #    access_roles entry spans every department.
    #
    # 2. The free-text department PARSED FROM THE QUERY (metadata_filters,
    #    from query_metadata_parser) matches on broad keyword lists
    #    ("financial" -> Finance) and would false-positive on a Legal
    #    document that merely mentions "financial obligations". Ranking
    #    handles department relevance; the filter stays a security
    #    boundary only.
    # --------------------------------------------------

    return qmodels.Filter(must=must)

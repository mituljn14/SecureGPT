"""
Extract metadata constraints from the user's query.

The parser extracts things like

- report name
- quarter
- year
- month
- department

These are later converted into Qdrant filters.

This parser DOES NOT call Qdrant.
"""

import re

# Add every report type that exists in your dataset.
REPORT_NAMES = [

    "Financial Forecast Report",
    "Budget Tracker",
    "Procurement Performance Report",
    "Vendor Management Report",
    "Risk Assessment Report",
    "Compliance Audit Report",
]

DEPARTMENT_MAPPING = {
    "Executive_Corporate": ["executive", "corporate", "chairman", "shareholder", "shareholders", "annual report", "ceo", "board"],
    "Operations": ["operations", "manufacturing", "plant", "production", "coo", "logistics"],
    "Finance": ["finance", "budget", "financial", "cfo", "accounting"],
    "Technology_Digital": ["technology", "digital", "it", "software", "engineering", "tech", "cio", "cto", "ciso", "cdo"],
    "Sales_Commercial": ["sales", "commercial", "revenue", "marketing", "cmo", "cco"],
    "Legal_Compliance": ["legal", "compliance", "audit", "law", "clo", "regulatory"],
    "People_Culture": ["hr", "people", "culture", "employee", "human resources", "chro", "hiring"]
}

MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
    "Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Sept","Oct","Nov","Dec"
]


def parse_query_metadata(query: str):

    metadata = {}

    query_lower = query.lower()

    # -------------------------------------------------------
    # Report Name
    # -------------------------------------------------------

    for report in REPORT_NAMES:

        if report.lower() in query_lower:

            metadata["report_name"] = report

            break

    # -------------------------------------------------------
    # Department
    # -------------------------------------------------------
    for dept, keywords in DEPARTMENT_MAPPING.items():
        if any(re.search(rf"\b{re.escape(k)}\b", query_lower) for k in keywords):
            metadata["department"] = dept
            break

    # -------------------------------------------------------
    # Quarter
    # -------------------------------------------------------
    q = re.search(r"\bQ([1-4])\b", query, re.IGNORECASE)
    if q:
        metadata["quarter"] = int(q.group(1))

    # -------------------------------------------------------
    # Year -- range first ("2014-2018", "2014 to 2018", "between 2014
    # and 2018"), single year second. This ordering matters: a range
    # query matched by the single-year pattern alone would grab only the
    # FIRST year (e.g. "2014" out of "2014-2018") and, if applied as a
    # strict equality filter, would wrongly exclude every other year in
    # the range being asked about -- exactly what happened before this
    # fix ("total revenue from 2014-2018" returned zero results because
    # it filtered to fiscal_year==2014 only, excluding the 2015-2018 data
    # needed to answer the question).
    # -------------------------------------------------------
    yr = re.search(r"\b(20\d{2})\s*(?:-|–|to|through)\s*(20\d{2})\b", query, re.IGNORECASE)
    if yr:
        lo, hi = int(yr.group(1)), int(yr.group(2))
        metadata["year_range"] = (min(lo, hi), max(lo, hi))
    else:
        y = re.search(r"\b(20\d{2})\b", query)
        if y:
            metadata["year"] = int(y.group(1))

    # -------------------------------------------------------
    # Month
    # -------------------------------------------------------

    for month in MONTHS:

        if month.lower() in query_lower:

            metadata["month"] = month

            break

    return metadata
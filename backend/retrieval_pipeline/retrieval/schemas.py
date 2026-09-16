from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class RetrievedChunk:
    """
    One retrieved chunk, carrying everything the context formatter and the
    coding agent need. Field names match the v2 Qdrant payload exactly
    (see pipeline/chunking/metadata.py in the ingestion project) --
    previously this dataclass only exposed 5 fields (text/department/
    file_name/page/heading) and `heading` didn't even exist in the payload
    (the real field is heading_path + section_title), so every chunk's
    heading rendered as None and ~34 useful metadata fields were silently
    dropped before the LLM ever saw them.
    """

    score: float
    text: str

    # identity / provenance
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    parent_block_id: Optional[str] = None
    source_path: Optional[str] = None
    file_name: Optional[str] = None
    source_format: Optional[str] = None
    doc_title: Optional[str] = None
    content_hash: Optional[str] = None

    # access control
    department: Optional[str] = None
    sub_department: Optional[str] = None
    access_roles: List[str] = field(default_factory=list)
    sensitivity: Optional[str] = None

    # navigation / structure
    functional_area: Optional[str] = None
    region: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    section_title: Optional[str] = None
    chunk_type: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None

    # citation locators
    page: Optional[int] = None
    sheet: Optional[str] = None
    slide: Optional[int] = None
    row_range: Optional[List[int]] = None
    row_total: Optional[int] = None
    table_columns: Optional[List[str]] = None
    table_caption: Optional[str] = None

    # context recovery
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # temporal
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    fiscal_half: Optional[int] = None
    doc_date: Optional[str] = None

    # retrieval aids
    token_count: Optional[int] = None
    has_table: bool = False
    identifiers: List[str] = field(default_factory=list)
    context_header: Optional[str] = None

    # raw payload, kept for anything not surfaced above (e.g. new fields
    # added to the ingestion pipeline later without a schema change here)
    payload: Dict[str, Any] = field(default_factory=dict)

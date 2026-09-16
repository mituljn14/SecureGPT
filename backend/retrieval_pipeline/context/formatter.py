from retrieval_pipeline.retrieval.schemas import RetrievedChunk


def _citation_line(chunk: RetrievedChunk) -> str:
    """One-line citation locator, format depends on what the chunk actually
    has (page for PDF, sheet for XLSX, slide for PPTX, row range for a
    table slice) -- these fields were captured at ingest specifically so
    citations can be exact instead of "somewhere in this file"."""
    parts = []
    if chunk.fiscal_year:
        period = f"FY{chunk.fiscal_year}"
        if chunk.fiscal_quarter:
            period += f" Q{chunk.fiscal_quarter}"
        parts.append(period)
    if chunk.page:
        parts.append(f"page {chunk.page}")
    if chunk.sheet:
        parts.append(f"sheet '{chunk.sheet}'")
    if chunk.slide:
        parts.append(f"slide {chunk.slide}")
    if chunk.row_range:
        lo, hi = chunk.row_range[0], chunk.row_range[1]
        span = f"row {lo}" if lo == hi else f"rows {lo}-{hi}"
        if chunk.row_total:
            span += f" of {chunk.row_total}"
        parts.append(span)
    return "  |  ".join(parts)


def format_chunk(chunk: RetrievedChunk) -> str:
    section = " > ".join(chunk.heading_path) if chunk.heading_path else (chunk.section_title or "-")
    citation = _citation_line(chunk)

    header_lines = [
        f"Document    : {chunk.file_name}",
        f"Department  : {chunk.department}" + (f" / {chunk.sub_department}" if chunk.sub_department else ""),
        f"Section     : {section}",
    ]
    if citation:
        header_lines.append(f"Location    : {citation}")

    # Table chunks carry structural metadata prose chunks don't -- surface
    # it explicitly rather than relying on the LLM to infer it from raw
    # markdown pipe syntax.
    if chunk.chunk_type == "table_card":
        header_lines.append(f"Type        : Table summary — {chunk.table_caption or 'unnamed table'}")
        if chunk.table_columns:
            header_lines.append(f"Columns     : {', '.join(chunk.table_columns)}")
    elif chunk.chunk_type in ("table_rows", "table_record"):
        caption = chunk.table_caption or chunk.section_title
        header_lines.append(f"Type        : Table data" + (f" — {caption}" if caption else ""))

    if chunk.identifiers:
        header_lines.append(f"Identifiers : {', '.join(chunk.identifiers[:8])}")

    return "\n".join(header_lines) + f"\n\nContent:\n{chunk.text}\n"

# pipeline/converters/table_md.py
"""
Shared helper: render rows of data as a *valid* markdown table (header +
--- separator + rows). Every tabular converter (xlsx/csv/docx-table/
pptx-table) uses this so table output is consistent and something a
markdown-table detector downstream can reliably recognize.
"""
from typing import List, Optional, Sequence


def _clean_cell(value) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    # escape pipe characters so they don't break the table grid
    text = text.replace("|", "\\|")
    return text


def rows_to_markdown_table(
    header: Optional[Sequence] ,
    rows: List[Sequence],
    caption: Optional[str] = None,
) -> str:
    """
    header: list of column names (or None -> synthesize Col1, Col2, ...)
    rows: list of row tuples/lists
    caption: optional one-line title placed above the table (helps
             embedding relevance — a bare table has no context otherwise).
    """
    if not rows:
        return ""

    n_cols = len(header) if header else max((len(r) for r in rows), default=0)
    if n_cols == 0:
        return ""

    if not header:
        header = [f"Col{i+1}" for i in range(n_cols)]

    out = []
    if caption:
        out.append(f"**{caption}**\n")

    out.append("| " + " | ".join(_clean_cell(h) for h in header) + " |")
    out.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")

    for row in rows:
        cells = list(row) + [""] * (n_cols - len(row))  # pad short rows
        cells = cells[:n_cols]  # truncate long rows
        out.append("| " + " | ".join(_clean_cell(c) for c in cells) + " |")

    return "\n".join(out)

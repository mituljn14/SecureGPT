# pipeline/converters/csv_to_md.py
"""CSV -> Markdown: header + real markdown table for all data rows.
Row-group chunking (for very large CSVs) happens later, in the chunker —
this stage just produces one faithful, complete markdown table."""
import csv

from .table_md import rows_to_markdown_table


def convert(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return ""

    header, body = rows[0], rows[1:]
    return rows_to_markdown_table(header, body)

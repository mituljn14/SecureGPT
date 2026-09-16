# pipeline/chunking/tables.py
"""
Stage 2a — table routing.

A table NEVER enters the prose splitter. The old pipeline ran
pipe-delimited tables through a sentence regex, which fragmented rows at
every decimal point and dropped the header row -- the root cause of the
table-retrieval failures this pipeline exists to fix.

Every table emits:
  1. a `table_card` chunk  -- caption + column names + row count + sample
     rows, so schema-level questions ("which report tracks EBITDA margin
     by quarter") retrieve the TABLE rather than an arbitrary row slice.
  2. data chunks, routed by table SHAPE:

     whole table   (fits budget)          -> one chunk
     entity table  (registry of records)  -> 1 record per chunk, rendered
                                             vertically as "Column: value"
     value table   (measurements/log)     -> row groups sized by tokens,
                                             header repeated in each slice

Why shape matters, measured on this corpus: the largest CSVs
(Enterprise_Dashboard_Catalog, 5,001 x 20) have rows that are 88%
interleaved by business domain -- adjacent rows share no topic. Grouping
them produces a chunk whose single vector is the average of ~7 unrelated
domains, which matches nothing strongly. Each row, by contrast, is a
complete self-describing record. So registry rows are chunked one per
record ("one object = one chunk"), while genuine measurement logs, whose
rows only mean something in aggregate, keep row-group slicing.
"""
from typing import Dict, List

from .blocks import Block
from .tokens import count_tokens

TABLE_WHOLE_MAX = 700       # a table at or under this stays in one piece
ROW_GROUP_TARGET = 500      # token budget per row-group slice
ENTITY_MIN_COLS = 8         # narrower than this is a value table, not a registry
ENTITY_MIN_ROWS = 10
ENTITY_ID_UNIQUENESS = 0.9  # first column must be ~unique to be an ID
SAMPLE_ROWS_IN_CARD = 3


def _render_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        cells = (list(r) + [""] * len(header))[: len(header)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _render_record(header: List[str], row: List[str]) -> str:
    """Vertical 'Column: value' rendering. Keeps every value bound to its
    column name, which is what survives retrieval when a single record is
    pulled out of a 20-column table."""
    cells = (list(row) + [""] * len(header))[: len(header)]
    return "\n".join(f"{h}: {v}" for h, v in zip(header, cells) if str(v).strip())


def _is_numeric(v: str) -> bool:
    s = str(v).strip().replace(",", "").replace("%", "").replace("-", "", 1).replace(".", "", 1)
    return s.isdigit() and s != ""


def classify_table(header: List[str], rows: List[List[str]]) -> str:
    """'entity' (registry of independent records) vs 'value' (measurement
    or time-series log whose rows only mean something together)."""
    if len(header) < ENTITY_MIN_COLS or len(rows) < ENTITY_MIN_ROWS:
        return "value"

    first = [r[0] for r in rows if r]
    if not first:
        return "value"
    if len(set(first)) / len(first) < ENTITY_ID_UNIQUENESS:
        return "value"

    # A wide table of mostly numbers is a measurement grid, not a registry,
    # even when its first column is a unique point/checkpoint id.
    sample = rows[: min(50, len(rows))]
    numeric_cells = sum(
        1 for r in sample for v in r[1:] if _is_numeric(v)
    )
    total_cells = sum(len(r[1:]) for r in sample) or 1
    if numeric_cells / total_cells > 0.5:
        return "value"

    return "entity"


def chunk_table(block: Block, table_index: int) -> List[Dict]:
    """Returns a list of raw chunk dicts:
    {text, chunk_type, locator:{...}} -- metadata/context is added later."""
    header = block.header or []
    rows = block.rows or []
    if not header:
        return []

    caption = block.caption or (block.heading_path[-1] if block.heading_path else "Table")
    n_rows = len(rows)
    chunks: List[Dict] = []

    # ---- 1. table card (always) -------------------------------------
    sample = rows[:SAMPLE_ROWS_IN_CARD]
    card_lines = [
        f"Table: {caption}",
        f"Columns ({len(header)}): " + ", ".join(header),
        f"Rows: {n_rows}",
    ]
    if sample:
        card_lines.append("Sample rows:")
        card_lines.append(_render_table(header, sample))
    chunks.append({
        "text": "\n".join(card_lines),
        "chunk_type": "table_card",
        "locator": {"table_index": table_index, "row_total": n_rows,
                    "table_columns": header, "caption": caption},
    })

    if not rows:
        return chunks

    # ---- 2. data chunks ---------------------------------------------
    whole = _render_table(header, rows)
    if count_tokens(whole) <= TABLE_WHOLE_MAX:
        chunks.append({
            "text": whole,
            "chunk_type": "table_rows",
            "locator": {"table_index": table_index, "row_range": [1, n_rows],
                        "row_total": n_rows, "table_columns": header, "caption": caption},
        })
        return chunks

    shape = classify_table(header, rows)

    if shape == "entity":
        for idx, row in enumerate(rows, start=1):
            rec = _render_record(header, row)
            if not rec.strip():
                continue
            chunks.append({
                "text": rec,
                "chunk_type": "table_record",
                "locator": {"table_index": table_index, "row_range": [idx, idx],
                            "row_total": n_rows, "table_columns": header,
                            "caption": caption},
            })
        return chunks

    # value table -> row groups sized by tokens, never splitting a row
    head_cost = count_tokens("| " + " | ".join(header) + " |") * 2
    group: List[List[str]] = []
    group_start = 1
    budget = head_cost

    def flush_group(end_idx: int):
        if not group:
            return
        chunks.append({
            "text": _render_table(header, group),
            "chunk_type": "table_rows",
            "locator": {"table_index": table_index,
                        "row_range": [group_start, end_idx],
                        "row_total": n_rows, "table_columns": header,
                        "caption": caption},
        })

    for idx, row in enumerate(rows, start=1):
        row_tokens = count_tokens("| " + " | ".join(str(c) for c in row) + " |")
        if group and budget + row_tokens > ROW_GROUP_TARGET:
            flush_group(idx - 1)
            group = []
            group_start = idx
            budget = head_cost
        group.append(row)
        budget += row_tokens

    flush_group(n_rows)
    return chunks

# pipeline/chunking/chunker.py
"""
Orchestrator: one markdown file -> a list of finished chunk payloads.

Ties the stages together:
    normalize -> parse blocks -> route by type -> enrich -> link
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..converters.md_normalize import normalize_tables
from .blocks import Block, parse_blocks
from .metadata import (build_context_header, build_payload, detect_region,
                       detect_sensitivity, extract_identifiers, fiscal_from_date,
                       make_block_id, make_chunk_id, make_doc_id, parse_doc_date,
                       parse_fiscal, resolve_access)
from .prose import MIN_TOKENS, chunk_prose
from .tables import chunk_table
from .tokens import count_tokens

_H1_RE = re.compile(r"^#\s+(.*)$", re.M)
_SHEET_RE = re.compile(r"^Sheet:\s*(.+)$", re.I)
_SLIDE_RE = re.compile(r"^Slide\s+(\d+)", re.I)
_PAGE_RE = re.compile(r"page\s*(\d+)", re.I)


def _doc_title(text: str, file_name: str) -> str:
    m = _H1_RE.search(text)
    if m:
        t = m.group(1).strip().strip("*").strip()
        if t:
            return t
    return Path(file_name).stem


_MD_NOISE = re.compile(r"[*_`#>|\-\s]+")

MIN_SUBSTANTIVE_CHARS = 15


def _is_substantive(chunk: Dict) -> bool:
    """Reject chunks that would embed to nothing useful.

    Catches the leftover-caption case ("**Interim_Statements**" alone),
    stray separator rows, and bodies that merely repeat the heading they
    sit under -- all of which cost a vector and can only pollute results.
    Table chunks are exempt from the length floor: a short table is still
    real data.
    """
    body = chunk.get("text", "").strip()
    if not body:
        return False
    if chunk.get("chunk_type", "").startswith("table"):
        return True

    stripped = _MD_NOISE.sub(" ", body).strip()
    if len(stripped) < MIN_SUBSTANTIVE_CHARS:
        return False

    # body adds nothing over the heading it lives under
    trail = [h for h in chunk.get("heading_path", []) if h]
    if trail:
        norm = lambda s: _MD_NOISE.sub("", s).lower()
        if norm(body) == norm(trail[-1]):
            return False
    return True


def _locator_from_headings(heading_path: List[str]) -> Dict:
    """Recover sheet / slide / page locators from the heading trail our
    converters emit ('## Sheet: Business Unit Variance', '## Slide 4: ...')."""
    loc: Dict = {}
    for h in heading_path:
        if not h:
            continue
        m = _SHEET_RE.match(h.strip())
        if m:
            loc["sheet"] = m.group(1).strip()
        m = _SLIDE_RE.match(h.strip())
        if m:
            loc["slide"] = int(m.group(1))
    return loc


def chunk_document(*, md_path: Path, markdown_root: Path, corpus_root: Path,
                   departments: Dict) -> List[Dict]:
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    text = normalize_tables(raw)

    rel_md = md_path.relative_to(markdown_root)
    # markdown/<...>/report.pdf.md  ->  <...>/report.pdf
    source_rel = rel_md.parent / rel_md.name[:-3]
    source_path = str(source_rel).replace("\\", "/")
    file_name = source_rel.name
    source_format = Path(file_name).suffix.lower().lstrip(".")

    parts = list(source_rel.parts)
    department = parts[0] if parts else "Unknown"
    sub_department = parts[1] if len(parts) > 1 else ""
    functional_area = parts[2] if len(parts) > 3 else None

    access_roles = resolve_access(departments, department, sub_department)
    region = detect_region(parts)
    fiscal = parse_fiscal(source_path)
    doc_date = parse_doc_date(text[:4000]) or None
    sensitivity = detect_sensitivity(text)
    doc_title = _doc_title(text, file_name)
    doc_id = make_doc_id(source_path)

    blocks: List[Block] = parse_blocks(text)

    # ---- route every block to raw chunks -------------------------------
    raw_chunks: List[Dict] = []
    table_counter = 0
    for b_idx, block in enumerate(blocks):
        block_id = make_block_id(source_path, b_idx)
        base_loc = _locator_from_headings(block.heading_path)

        if block.type == "table":
            table_counter += 1
            for c in chunk_table(block, table_counter):
                c["locator"] = {**base_loc, **c.get("locator", {})}
                c["heading_path"] = block.heading_path
                c["block_id"] = block_id
                raw_chunks.append(c)

        elif block.type == "code":
            raw_chunks.append({
                "text": block.text, "chunk_type": "code",
                "locator": base_loc, "heading_path": block.heading_path,
                "block_id": block_id,
            })

        else:  # prose
            pieces = chunk_prose(block.text)
            for piece in pieces:
                raw_chunks.append({
                    "text": piece, "chunk_type": "prose",
                    "locator": base_loc, "heading_path": block.heading_path,
                    "block_id": block_id,
                })

    # ---- drop chunks that carry no information --------------------------
    raw_chunks = [c for c in raw_chunks if _is_substantive(c)]
    if not raw_chunks:
        return []

    # ---- build payloads, then link neighbours ---------------------------
    total = len(raw_chunks)
    ids = [make_chunk_id(source_path, i, 0) for i in range(total)]

    payloads: List[Dict] = []
    for i, c in enumerate(raw_chunks):
        loc = c.get("locator", {})
        body = c["text"].strip()

        # Per-chunk temporal resolution: a registry/log row usually carries its
        # own date even when the filename has none, so fall back to the chunk's
        # content before giving up on fiscal_year.
        chunk_date = parse_doc_date(body)
        chunk_fiscal = dict(fiscal)
        if chunk_fiscal.get("fiscal_year") is None and chunk_date:
            chunk_fiscal = fiscal_from_date(chunk_date)

        # Captions belong to TABLES only -- adding one to a prose chunk just
        # repeats the heading it already sits under ("... › Chairman's Letter
        # › Annual Report 2018"). A bare "Table" caption (CSVs have no heading
        # to borrow) carries no meaning, so fall back to the document title.
        caption = None
        if c["chunk_type"].startswith("table"):
            caption = loc.get("caption")
            if not caption or caption.strip().lower() in ("table", "sheet1", "sheet"):
                caption = doc_title.replace("_", " ").strip() or None

        ctx = build_context_header(
            department=department, sub_department=sub_department,
            functional_area=functional_area, file_name=file_name,
            fiscal_year=chunk_fiscal.get("fiscal_year"),
            fiscal_quarter=chunk_fiscal.get("fiscal_quarter"),
            heading_path=c.get("heading_path", []),
            caption=caption,
            row_range=loc.get("row_range"), row_total=loc.get("row_total"),
        )
        embed_text = f"{ctx}\n———\n{body}" if ctx else body

        payloads.append(build_payload(
            chunk_id=ids[i], doc_id=doc_id, block_id=c["block_id"],
            source_path=source_path, file_name=file_name,
            source_format=source_format, doc_title=doc_title,
            department=department, sub_department=sub_department,
            functional_area=functional_area, access_roles=access_roles,
            sensitivity=sensitivity, region=region,
            heading_path=[h for h in c.get("heading_path", []) if h],
            chunk_type=c["chunk_type"], chunk_index=i, total_chunks=total,
            locator=loc, fiscal=chunk_fiscal, doc_date=chunk_date or doc_date,
            text=body, context_header=ctx, token_count=count_tokens(embed_text),
            identifiers=extract_identifiers(f"{file_name}\n{body}"),
            prev_chunk_id=ids[i - 1] if i > 0 else None,
            next_chunk_id=ids[i + 1] if i < total - 1 else None,
        ))
        payloads[-1]["embed_text"] = embed_text

    return payloads

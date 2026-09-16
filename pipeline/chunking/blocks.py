# pipeline/chunking/blocks.py
"""
Stage 1 — structural parse.

Turns a markdown document into a flat list of typed blocks, each carrying
the heading path it lives under. This is what makes "structure first,
tokens second" possible: the chunker never sees raw text, only blocks
that already know whether they are prose, a table, or code.

Design notes:
  - A table block keeps its parsed header + rows (not just raw text) so
    the table chunker can re-render slices with a repeated header and
    decide entity-vs-timeseries shape without re-parsing.
  - A bold single line immediately preceding a table is captured as that
    table's caption (this is exactly what xlsx_to_md.py emits), so the
    caption travels with every slice of the table instead of becoming an
    orphaned 4-token prose chunk.
  - An unclosed code fence terminates at the next heading rather than
    swallowing the rest of the document. One file in the corpus
    (NOVA-HR-TA-POL-001_Recruitment_Policy) has a stray ```mermaid inside
    a heading; without this guard it would become one giant chunk.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEP_ROW_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
FENCE_RE = re.compile(r"^\s*```")
BOLD_LINE_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")


@dataclass
class Block:
    type: str                                   # prose | table | code
    text: str = ""
    heading_path: List[str] = field(default_factory=list)
    caption: Optional[str] = None
    header: Optional[List[str]] = None          # table only
    rows: Optional[List[List[str]]] = None      # table only


def _split_cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


def parse_blocks(text: str) -> List[Block]:
    lines = text.split("\n")
    blocks: List[Block] = []
    heading_stack: List[str] = []
    prose_buf: List[str] = []
    i = 0

    def flush_prose():
        nonlocal prose_buf
        body = "\n".join(prose_buf).strip()
        if body:
            blocks.append(Block(type="prose", text=body, heading_path=list(heading_stack)))
        prose_buf = []

    def take_pending_caption() -> Optional[str]:
        """If the immediately preceding prose was a lone **bold line**,
        pull it off and use it as the upcoming table's caption.

        Trailing blank lines are dropped first: converters emit
        "**Caption**\n\n| table |", so without this the buffer ends in ""
        and the caption is never captured -- it then survives as its own
        content-free prose chunk ("**Interim_Statements**" and nothing else).
        """
        while prose_buf and not prose_buf[-1].strip():
            prose_buf.pop()
        if prose_buf:
            m = BOLD_LINE_RE.match(prose_buf[-1].strip())
            if m:
                prose_buf.pop()
                return m.group(1).strip()
        if blocks and blocks[-1].type == "prose":
            last_line = blocks[-1].text.split("\n")[-1].strip()
            m = BOLD_LINE_RE.match(last_line)
            if m:
                remaining = "\n".join(blocks[-1].text.split("\n")[:-1]).strip()
                if remaining:
                    blocks[-1].text = remaining
                else:
                    blocks.pop()
                return m.group(1).strip()
        return None

    while i < len(lines):
        line = lines[i]

        # ---- heading ----
        m = HEADING_RE.match(line)
        if m:
            flush_prose()
            level = len(m.group(1))
            title = m.group(2).strip().strip("*").strip()
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title)
            i += 1
            continue

        # ---- code fence ----
        if FENCE_RE.match(line):
            flush_prose()
            fence_lines = [line]
            i += 1
            while i < len(lines):
                if FENCE_RE.match(lines[i]):
                    fence_lines.append(lines[i])
                    i += 1
                    break
                if HEADING_RE.match(lines[i]):     # unclosed fence guard
                    break
                fence_lines.append(lines[i])
                i += 1
            body = "\n".join(fence_lines).strip()
            if body:
                blocks.append(Block(type="code", text=body, heading_path=list(heading_stack)))
            continue

        # ---- table ----
        if (TABLE_ROW_RE.match(line) and not SEP_ROW_RE.match(line)
                and i + 1 < len(lines) and SEP_ROW_RE.match(lines[i + 1])):
            caption = take_pending_caption()
            flush_prose()
            header = _split_cells(line)
            i += 2                                   # skip header + separator
            rows: List[List[str]] = []
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                if not SEP_ROW_RE.match(lines[i]):
                    rows.append(_split_cells(lines[i]))
                i += 1
            blocks.append(Block(
                type="table",
                heading_path=list(heading_stack),
                caption=caption,
                header=header,
                rows=rows,
            ))
            continue

        prose_buf.append(line)
        i += 1

    flush_prose()
    return blocks

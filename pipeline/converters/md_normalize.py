# pipeline/converters/md_normalize.py
"""
Markdown table repair / normalization.

Some source .md files in the corpus (Enterprise Applications/M365,
Oracle, SAP "output_phase*" folders) contain structurally broken tables
that markdown parsers -- and our chunker's table detector -- cannot read:

    | Activity | Business Owner | Solution Architect | Dev Lead | QA Lead | CAB |
    |--- |
    |--- |
    |--- |
    |--- |
    |--- |
    |---|

The separator was emitted one cell per LINE instead of a single row with
N cells. Left alone, a table detector sees a 6-column header followed by
six 1-column "rows", so the table never gets recognised as a table, and
its rows fall through to the prose splitter -- exactly the failure mode
this pipeline exists to prevent.

normalize_tables() rebuilds a correct separator row of the right width
and pads/truncates ragged body rows to the header's column count, so
every table in markdown/ is a well-formed grid before chunking.
"""
import re
from typing import List

TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEP_ROW_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def _split_cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return re.split(r"(?<!\\)\|", s)


def _render_row(cells: List[str]) -> str:
    return "| " + " | ".join(c.strip() for c in cells) + " |"


_STRIP_MD = re.compile(r"\*\*|<br\s*/?>")
# a lowercase letter or closing bracket butting straight against a capital
# letter marks where two column names were fused: "MarketAnnual" -> Market|Annual
_FUSED_BOUNDARY = re.compile(r"(?<=[a-z\)\]])(?=[A-Z])")


def _repair_header_cells(cells: List[str]) -> List[str]:
    """Repair PDF table headers whose column names were split at the wrong
    cell boundaries by layout detection, e.g.

        ['Client Program ID', 'Business Unit', 'MarketAnn',
         'ual Contract Value (INR Lakh', ')Relationship Tenure (Years)']

    should be

        ['Client Program ID', 'Business Unit', 'Market',
         'Annual Contract Value (INR Lakh)', 'Relationship Tenure (Years)']

    All the text is present -- only the boundaries are wrong -- so we glue
    the broken boundaries back together and re-split on fused word joins.
    Applied ONLY if the repair yields exactly the original column count, so
    a failed repair can never desynchronise the header from the data rows.
    """
    n = len(cells)
    plain = [_STRIP_MD.sub("", c).strip() for c in cells]
    if not any(plain):
        return cells

    # 1. glue boundaries that cut through a word
    glued: List[str] = []
    for cell in plain:
        if glued and glued[-1] and cell:
            prev, cur = glued[-1], cell
            if (prev[-1].isalpha() and cur[0].islower()) or cur[0] in ")]}":
                glued[-1] = prev + cur
                continue
        glued.append(cell)

    if len(glued) == n:
        return cells  # nothing was broken

    # 2. re-split on fused joins ("...Lakh)Relationship" -> two names)
    rebuilt: List[str] = []
    for cell in glued:
        rebuilt.extend(p.strip() for p in _FUSED_BOUNDARY.split(cell) if p.strip())

    # 3. only accept a repair that restores the exact column count
    return rebuilt if len(rebuilt) == n else cells


def normalize_tables(text: str) -> str:
    lines = text.split("\n")
    out: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # a table starts at a pipe row that is NOT itself a separator
        if TABLE_ROW_RE.match(line) and not SEP_ROW_RE.match(line):
            header_cells = _split_cells(line)
            n_cols = len(header_cells)

            # consume any run of separator lines immediately after the header
            j = i + 1
            saw_sep = False
            while j < len(lines) and SEP_ROW_RE.match(lines[j]):
                saw_sep = True
                j += 1

            if not saw_sep:
                out.append(line)
                i += 1
                continue

            # collect body rows
            body: List[List[str]] = []
            while j < len(lines) and TABLE_ROW_RE.match(lines[j]):
                if SEP_ROW_RE.match(lines[j]):
                    j += 1
                    continue
                body.append(_split_cells(lines[j]))
                j += 1

            out.append(_render_row(_repair_header_cells(header_cells)))
            out.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
            for row in body:
                if len(row) < n_cols:
                    row = row + [""] * (n_cols - len(row))
                elif len(row) > n_cols:
                    # merge the overflow into the last column rather than
                    # dropping data (ragged rows from PDF table detection)
                    row = row[: n_cols - 1] + [" ".join(c.strip() for c in row[n_cols - 1:])]
                out.append(_render_row(row))

            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out)

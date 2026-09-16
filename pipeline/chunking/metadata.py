# pipeline/chunking/metadata.py
"""
Metadata extraction + chunk payload construction.

Everything here exists to make retrieval filterable and context
recoverable. Three groups matter most:

  RBAC      department / sub_department / access_roles  -- enforced
            server-side in Qdrant, never client-side.
  TEMPORAL  fiscal_year / fiscal_quarter as INTEGERS. The old pipeline
            filtered years with MatchText(file_name, "2027"), which also
            matches "20271" and "FY2027_v2". Parsed integers make range
            and equality queries exact.
  RECOVERY  doc_id / parent_block_id / prev_chunk_id / next_chunk_id --
            the handoff points to fetch neighbours, reassemble a whole
            table from its slices, or load the entire source document.
"""
import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PIPELINE_VERSION = "2.0.0"
NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

# ---------------------------------------------------------------- fiscal time
_FY_PATTERNS = [
    (re.compile(r"Q([1-4])\s*[_\-]?\s*FY\s*[_\-]?\s*(20\d{2})", re.I), "qfy"),
    (re.compile(r"FY\s*[_\-]?\s*(20\d{2})\s*[_\-]?\s*Q([1-4])", re.I), "fyq"),
    (re.compile(r"(?<!\d)(20\d{2})\s*Q([1-4])(?!\d)"), "yq"),
    (re.compile(r"(?<![A-Za-z0-9])Q([1-4])\s*[_\-\s]\s*(20\d{2})(?!\d)"), "qy"),
    (re.compile(r"FY\s*[_\-]?\s*(20\d{2})", re.I), "fy"),
    (re.compile(r"(?<![A-Za-z0-9])H([12])\s*[_\-]?\s*(20\d{2})(?!\d)", re.I), "hy"),
    (re.compile(r"(20\d{2})\s*W\d{1,2}\b", re.I), "yw"),
]
# NOTE: cannot use \b around a year here -- "_" is a regex word character, so
# \b never fires in "Annual_Report_2018.pdf" and the year is missed. Explicit
# digit lookarounds treat underscores/letters as boundaries correctly.
_BARE_YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")
_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_DMY_DATE = re.compile(r"\b(\d{1,2})[-/]([A-Z][a-z]{2})[-/](20\d{2})\b")

# document/reference codes: FIN-FPA-RPT-001, DB-01-000001, NOVA-HR-TA-POL-001
_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{1,9}){1,5}\b")

_SENSITIVITY = [
    ("restricted", re.compile(r"\brestricted\b", re.I)),
    ("confidential", re.compile(r"\bconfidential\b|\bstrictly private\b", re.I)),
    ("internal", re.compile(r"\binternal use only\b|\binternal only\b", re.I)),
    ("public", re.compile(r"\bpublic\b", re.I)),
]

_REGIONS = {
    "apac": "APAC", "emea": "EMEA", "amer": "AMER", "americas": "AMER",
    "europe": "EMEA", "india": "India", "japan": "Japan", "australia": "Australia",
    "singapore": "Singapore", "south_korea": "South Korea", "china": "China",
    "germany": "Germany", "uk": "UK", "usa": "USA", "mexico": "Mexico",
    "sweden": "Sweden", "brazil": "Brazil",
}


def parse_fiscal(text: str) -> Dict[str, Optional[int]]:
    """Extract fiscal year/quarter from a path or filename. First match wins,
    patterns ordered most-specific first."""
    out = {"fiscal_year": None, "fiscal_quarter": None, "fiscal_half": None}
    for pat, kind in _FY_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if kind == "qfy":
            out["fiscal_quarter"], out["fiscal_year"] = int(m.group(1)), int(m.group(2))
        elif kind in ("fyq",):
            out["fiscal_year"], out["fiscal_quarter"] = int(m.group(1)), int(m.group(2))
        elif kind in ("yq", "yw"):
            out["fiscal_year"] = int(m.group(1))
            if kind == "yq":
                out["fiscal_quarter"] = int(m.group(2))
        elif kind == "qy":
            out["fiscal_quarter"], out["fiscal_year"] = int(m.group(1)), int(m.group(2))
        elif kind == "fy":
            out["fiscal_year"] = int(m.group(1))
        elif kind == "hy":
            out["fiscal_half"], out["fiscal_year"] = int(m.group(1)), int(m.group(2))
        break
    if out["fiscal_year"] is None:
        m = _BARE_YEAR.search(text)
        if m:
            out["fiscal_year"] = int(m.group(1))
    return out


def parse_doc_date(text: str) -> Optional[str]:
    m = _ISO_DATE.search(text)
    if m:
        return m.group(0)
    m = _DMY_DATE.search(text)
    if m:
        try:
            return datetime.strptime(m.group(0).replace("/", "-"), "%d-%b-%Y").strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def fiscal_from_date(date_str: str) -> Dict[str, Optional[int]]:
    """Derive year/quarter from an ISO date. Used when the file PATH carries
    no period but the chunk's own content does -- which is the norm for
    registry/log rows (e.g. a press-release record with 'Date: 2019-02-09'
    inside a master log whose filename has no year). Without this, temporal
    filtering is dead on ~99% of chunks."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return {"fiscal_year": None, "fiscal_quarter": None, "fiscal_half": None}
    return {
        "fiscal_year": dt.year,
        "fiscal_quarter": (dt.month - 1) // 3 + 1,
        "fiscal_half": 1 if dt.month <= 6 else 2,
    }


def extract_identifiers(text: str, limit: int = 25) -> List[str]:
    seen: List[str] = []
    for m in _IDENTIFIER.finditer(text):
        v = m.group(0)
        if v not in seen and not v.replace("-", "").isdigit():
            seen.append(v)
        if len(seen) >= limit:
            break
    return seen


def detect_sensitivity(text: str, default: str = "internal") -> str:
    head = text[:4000]
    for label, pat in _SENSITIVITY:
        if pat.search(head):
            return label
    return default


def detect_region(parts: List[str]) -> Optional[str]:
    for p in parts:
        key = p.strip("_").lower()
        if key in _REGIONS:
            return _REGIONS[key]
    return None


# ---------------------------------------------------------------- ids/hashes
def make_doc_id(source_path: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"doc::{source_path}"))


def make_block_id(source_path: str, block_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"block::{source_path}::{block_index}"))


def make_chunk_id(source_path: str, block_index: int, chunk_index: int) -> str:
    """Deterministic: re-ingesting the same file upserts the same points
    instead of duplicating. Includes the full relative PATH (not just the
    file name) so identically-named files in different folders of the
    same department cannot collide -- a bug in the old pipeline."""
    return str(uuid.uuid5(NAMESPACE, f"chunk::{source_path}::{block_index}::{chunk_index}"))


def content_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- departments
def load_departments(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_access(departments: Dict, department: str, sub_department: str) -> List[str]:
    """access_roles for a department/sub-department pair, with sensible
    fallbacks so a folder missing from departments.yaml still gets a
    restrictive (not permissive) role set."""
    dept_cfg = departments.get(department) or {}
    sub_cfg = dept_cfg.get(sub_department) or {}
    roles = sub_cfg.get("access_roles")
    if roles:
        return list(roles)
    for cfg in dept_cfg.values():
        if isinstance(cfg, dict) and cfg.get("access_roles"):
            pass
    return [sub_department or department, "CEO"]


# ---------------------------------------------------------------- context hdr
def build_context_header(*, department: str, sub_department: str, functional_area: Optional[str],
                         file_name: str, fiscal_year: Optional[int], fiscal_quarter: Optional[int],
                         heading_path: List[str], caption: Optional[str],
                         row_range: Optional[List[int]], row_total: Optional[int]) -> str:
    """Tier-1 contextual retrieval: a deterministic breadcrumb prepended to
    the text that gets EMBEDDED, so a chunk is self-describing even when
    retrieved in isolation. Costs nothing and needs no LLM."""
    line1 = " > ".join([p for p in (department, sub_department, functional_area) if p])

    period = ""
    if fiscal_year and fiscal_quarter:
        period = f" (FY{fiscal_year} Q{fiscal_quarter})"
    elif fiscal_year:
        period = f" (FY{fiscal_year})"
    line2 = f"{file_name}{period}"

    trail = [h for h in heading_path if h]
    if caption and (not trail or caption != trail[-1]):
        trail = trail + [caption]
    line3 = " › ".join(trail)
    # Row position is deliberately NOT embedded: "row 6 of 209" can never help
    # match a query, so it is pure token overhead across 600k+ record chunks.
    # It stays available in the payload (row_range / row_total) for citation.
    # A multi-row slice keeps its span, which does signal "partial table".
    if row_range and row_total and row_range[0] != row_range[1]:
        line3 = (line3 + " " if line3 else "") + f"(rows {row_range[0]}–{row_range[1]} of {row_total})"

    return "\n".join([l for l in (line1, line2, line3) if l])


def build_payload(*, chunk_id: str, doc_id: str, block_id: str, source_path: str,
                  file_name: str, source_format: str, doc_title: str,
                  department: str, sub_department: str, functional_area: Optional[str],
                  access_roles: List[str], sensitivity: str, region: Optional[str],
                  heading_path: List[str], chunk_type: str, chunk_index: int,
                  total_chunks: int, locator: Dict, fiscal: Dict, doc_date: Optional[str],
                  text: str, context_header: str, token_count: int,
                  identifiers: List[str], prev_chunk_id: Optional[str],
                  next_chunk_id: Optional[str]) -> Dict:
    return {
        # identity / provenance
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "parent_block_id": block_id,
        "source_path": source_path,
        "file_name": file_name,
        "source_format": source_format,
        "doc_title": doc_title,
        "content_hash": content_hash(text),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,

        # access control
        "department": department,
        "sub_department": sub_department,
        "access_roles": access_roles,
        "sensitivity": sensitivity,

        # navigation / structure
        "functional_area": functional_area,
        "region": region,
        "heading_path": heading_path,
        "section_title": heading_path[-1] if heading_path else None,
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,

        # citation locators
        "page": locator.get("page"),
        "sheet": locator.get("sheet"),
        "slide": locator.get("slide"),
        "row_range": locator.get("row_range"),
        "row_total": locator.get("row_total"),
        "table_columns": locator.get("table_columns"),
        "table_caption": locator.get("caption"),

        # context recovery
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,

        # temporal
        "fiscal_year": fiscal.get("fiscal_year"),
        "fiscal_quarter": fiscal.get("fiscal_quarter"),
        "fiscal_half": fiscal.get("fiscal_half"),
        "doc_date": doc_date,

        # retrieval aids
        "token_count": token_count,
        "has_table": chunk_type in ("table_rows", "table_record", "table_card"),
        "identifiers": identifiers,

        # content
        "text": text,
        "context_header": context_header,
    }

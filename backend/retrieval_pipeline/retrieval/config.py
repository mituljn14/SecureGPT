"""
Configuration for Retrieval Pipeline
"""
import os
from pathlib import Path

TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# --------------------------------------------------------------------------
# DENSE_RELEVANCE_THRESHOLD -- below this raw DENSE cosine similarity
# (single best match, dense vector only), treat a query as having no
# genuine corpus match. NOT the fused RRF score returned by hybrid search,
# which is purely rank-based (~1/(rank+k)) and gives a confident-looking
# ~0.5 to the #1 result EVEN WHEN NOTHING RELEVANT EXISTS, simply because
# something has to rank first -- checking "did retrieve() return any hits"
# has the same failure mode, since it always returns top_k results.
#
# Empirically measured on this corpus: a genuinely unrelated query landed
# ~0.45-0.46 dense cosine against its "best" (still irrelevant) match,
# while real corpus-matching queries landed 0.59-0.63+. 0.50 sits in that
# gap with margin on both sides. Shared by both the coding agent's
# retriever (coding_agent/retriever.py) and general RAG's fast-path
# confidence gate (graph/nodes.py) so the two don't drift out of sync.
# Tune if false positives/negatives are observed with more query volume --
# this is an empirical starting point, not a universal constant.
# --------------------------------------------------------------------------
DENSE_RELEVANCE_THRESHOLD = float(os.getenv("DENSE_RELEVANCE_THRESHOLD", "0.50"))

MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))

# Fast path is trusted without any LLM grading at or above this dense
# cosine; below it the query escalates to CRAG. Must stay >= the floor
# above (asserted in graph/workflow.py).
FAST_PATH_CONFIDENCE_THRESHOLD = float(os.getenv("FAST_PATH_CONFIDENCE_THRESHOLD", "0.55"))
# ---------------------------------------------------------------------------
# Deployment configuration -- single source of truth.
#
# The Qdrant URL and collection name were repeated as string literals across
# retrieve.py, qdrant_client.py, the supervisor, the coding agent and several
# telemetry labels. Changing deployment meant finding every copy, and a missed
# one fails silently: a query against the wrong collection still returns
# results, just meaningless ones. Everything now reads these.
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "securegpt_v2")
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "60"))

# Selecting a whole document to open and describe is a bigger commitment than
# adding a context chunk, so the vision agent needs a higher bar than
# DENSE_RELEVANCE_THRESHOLD. Measured on this corpus: a genuine document
# request scores ~0.77, while contentless phrasing ("make a bar chart of this
# data") still clears 0.50 at ~0.55. Tune via env as the corpus changes.
VISION_SOURCE_THRESHOLD = float(os.getenv("VISION_SOURCE_THRESHOLD", "0.65"))


# --------------------------------------------------------------------------
# CORPUS_ROOT -- the on-disk root that every chunk's `source_path` payload
# field is relative to (e.g. "Finance/CFO/.../report.xlsx" resolves to
# CORPUS_ROOT / "Finance/CFO/.../report.xlsx").
#
# This is what the coding agent mounts into its Docker sandbox for
# full-file analysis (e.g. "sum all 50,000 rows"), using source_path
# straight off the retrieved chunk's payload -- no separate ID/database
# layer required. See coding_agent/file_reader.py.
#
# Default assumes this file lives at
#   <SecureGPT>/backend/retrieval_pipeline/retrieval/config.py
# so parents[3] is the SecureGPT project root. Override with the
# CORPUS_ROOT env var if your corpus/ lives somewhere else.
# --------------------------------------------------------------------------
CORPUS_ROOT = Path(os.getenv("CORPUS_ROOT") or (Path(__file__).resolve().parents[3] / "corpus"))

# --------------------------------------------------------------------------
# MARKDOWN_ROOT -- the converted markdown mirror of CORPUS_ROOT (same
# relative paths, ".md" appended: "Finance/.../report.xlsx" ->
# "Finance/.../report.xlsx.md"). The coding agent reads THIS, not the raw
# source file, so it works from the exact same text that was chunked and
# embedded -- one canonical representation of every document instead of
# two (raw file vs. what's actually indexed) that could silently diverge.
# --------------------------------------------------------------------------
MARKDOWN_ROOT = Path(os.getenv("MARKDOWN_ROOT") or (Path(__file__).resolve().parents[3] / "markdown"))


# --------------------------------------------------------------------------
# Company identity -- who "our company" is. Read by the supervisor's
# routing prompt and by its deterministic company-reference rule (a query
# that names the organisation, or says "our"/"we"/"us", is a company
# question whatever a 3B classifier thinks). Override per deployment;
# COMPANY_ALIASES is comma-separated.
# --------------------------------------------------------------------------
COMPANY_NAME = os.getenv("COMPANY_NAME", "Nova Automobile Engineering Solutions Pvt. Ltd.")
COMPANY_ALIASES = [a.strip() for a in os.getenv(
    "COMPANY_ALIASES", "Nova Engineering, Nova, NAES"
).split(",") if a.strip()]


# --------------------------------------------------------------------------
# RBAC_SCOPE -- what a signed-in user may retrieve.
#
#   "role"        only chunks whose access_roles lists the user's role
#                 (each document is tagged with its owning sub-department
#                 role plus CEO, so a CIO cannot read a CTO document).
#   "department"  additionally every chunk whose `department` equals the
#                 user's own department -- colleagues share a department's
#                 documents, other departments stay invisible. CEO keeps
#                 cross-department reach via access_roles as before.
#
# Both are enforced inside the Qdrant filter, so the model never sees a
# chunk outside scope regardless of what the query asks for.
# --------------------------------------------------------------------------
#
# Default is "role": it enforces exactly what was decided at ingest time
# (least privilege, no query-time widening). Set RBAC_SCOPE=department to
# let colleagues share a department's documents.
RBAC_SCOPE = os.getenv("RBAC_SCOPE", "role").strip().lower()

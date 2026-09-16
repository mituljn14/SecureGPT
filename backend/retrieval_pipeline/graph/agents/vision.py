"""
Vision agent node -- multimodal document/image analysis.

Wraps vision_agent.core.pipeline.VisionPipeline so the supervisor can route
image / PDF / CAD / chart work to it. Retrieval inside the pipeline goes
through retrieval_pipeline (securegpt_v2, hybrid, RBAC-filtered), so the
caller's role is enforced here exactly as it is in enterprise chat.
"""
import os
import re

from retrieval_pipeline.graph.telemetry import log_event
from retrieval_pipeline.retrieval.config import QDRANT_URL
from retrieval_pipeline.llm.provider import get_network_target

# Built once -- the pipeline loads OCR and vision components eagerly.
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from vision_agent.core.pipeline import VisionPipeline
        _pipeline = VisionPipeline()
    return _pipeline


VISION_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif",
    ".pdf",
    # Office + plain text: read as text by the pipeline (see
    # vision_agent/document/text_extractor.py) so slides and documents can
    # be summarised and grounded against the corpus.
    ".pptx", ".ppt", ".docx", ".doc", ".txt", ".md", ".log", ".rtf",
}

# The pipeline also ingests tabular files (DocumentAnalyzer -> TABLE_DATA)
# and can chart them with its parameter-validated ChartGenerator -- no
# generated code, no sandbox. Charting a spreadsheet therefore belongs here
# rather than in the coding agent, which would write arbitrary matplotlib
# and execute it. Kept separate from VISION_EXTS so the supervisor only
# routes tabular files here when the request is actually visual.
TABULAR_EXTS = {".csv", ".xlsx", ".xls"}


INGESTIBLE_EXTS = VISION_EXTS | TABULAR_EXTS


def pick_vision_file(uploaded_files, include_tabular: bool = False):
    """First uploaded file the vision pipeline can handle.

    `include_tabular` is opt-in: spreadsheets only belong to vision when the
    user asked for something visual (a chart), otherwise they are the coding
    agent's job.
    """
    exts = INGESTIBLE_EXTS if include_tabular else VISION_EXTS
    for f in uploaded_files or []:
        if os.path.splitext(str(f))[1].lower() in exts:
            return str(f)
    return None


# Source-selection bar. Lives in retrieval/config.py (env-overridable via
# VISION_SOURCE_THRESHOLD) rather than as a second copy here.
from retrieval_pipeline.retrieval.config import VISION_SOURCE_THRESHOLD

# Vocabulary that carries no subject: verbs describing WHAT to do, words for
# the artifact itself, and demonstratives pointing at something supplied.
# Strip these and a request like "make a bar chart of this data" has nothing
# left -- it names no document, so there is nothing to go and retrieve.
_CONTENTLESS = {
    # actions
    "analyse", "analyze", "visualise", "visualize", "summarise", "summarize",
    "describe", "read", "explain", "chart", "plot", "graph", "show", "make",
    "create", "generate", "give", "draw", "extract", "process", "convert",
    "please", "can", "you", "me", "my",
    # artifact / generic nouns
    "data", "file", "files", "image", "document", "table", "attachment",
    "bar", "pie", "line", "histogram", "visualisation", "visualization",
    "chart's", "charts", "graphs", "plots",
    # demonstratives and filler
    "this", "that", "these", "those", "it", "its", "the", "a", "an", "of",
    "from", "for", "in", "on", "and", "with", "attached", "uploaded", "here",
    "above", "below", "please.", "is", "are", "be",
}


def wants_corpus_source(query: str) -> bool:
    """Whether a query with no attachment is actually asking about the corpus.

    "Visualise this data" names no subject: with nothing attached it cannot be
    answered by going and finding *some* document, and doing so would produce
    a confident analysis of a file the user never asked about.

    This is a cheap first filter; VISION_SOURCE_THRESHOLD above is the
    substantive gate.
    """
    words = re.findall(r"[a-z0-9']+", (query or "").lower())
    return any(w not in _CONTENTLESS for w in words)


def retrieve_corpus_file(query, role, department=None):
    """Find a corpus document this agent can ingest, for queries with no
    attachment ("chart the 2024 performance report").

    Retrieval is the same RBAC-filtered hybrid search the rest of the system
    uses, so the vision agent can only ever open documents the caller is
    cleared to read. Returns (absolute_path, chunk) or (None, None).
    """
    from pathlib import Path

    from retrieval_pipeline.graph.agents.coding_agent.retriever import Retriever
    from retrieval_pipeline.retrieval.config import CORPUS_ROOT
    from retrieval_pipeline.retrieval.embedding import embed_query_hybrid
    from retrieval_pipeline.retrieval.filter_builder import build_filter
    from retrieval_pipeline.retrieval.qdrant_client import get_client
    from retrieval_pipeline.retrieval.retrieve import COLLECTION_NAME

    if not wants_corpus_source(query):
        return None, None

    # Raw dense cosine against the stricter source-selection bar, NOT the RRF
    # score (rank-based: something always ranks first).
    dense, _sparse = embed_query_hybrid(query)
    probe = get_client().query_points(
        collection_name=COLLECTION_NAME, query=dense, using="dense",
        query_filter=build_filter(role=role, department=department, metadata_filters={}),
        limit=1,
    )
    if not probe.points or probe.points[0].score < VISION_SOURCE_THRESHOLD:
        return None, None

    retriever = Retriever()
    for chunk in retriever.search(query, role=role, top_k=12, department=department):
        source = chunk.get("source_path")
        if not source:
            continue
        if os.path.splitext(source)[1].lower() not in INGESTIBLE_EXTS:
            continue
        path = Path(source)
        if not path.is_absolute():
            path = Path(CORPUS_ROOT) / source
        if path.exists():
            return str(path), chunk
    return None, None


def vision_agent_node(state):
    print("=" * 80)
    print("Routed to: VISION AGENT")
    print("=" * 80)

    # Accept everything the pipeline can ingest, spreadsheets included.
    # `include_tabular` gates the supervisor's ROUTING decision (a .csv only
    # comes here when the request is visual); once routed, this node must
    # handle whatever it was sent. Re-applying the gate here made vision
    # reject the exact file it had just been routed for.
    target = pick_vision_file(state.get("uploaded_files"), include_tabular=True)
    source_chunk = None

    # No attachment? Find one in the corpus. Without this the agent could only
    # ever work on files the user had to locate and upload by hand, so a
    # question like "chart the 2024 performance report" was unanswerable even
    # though the document was sitting in the corpus.
    if not target:
        target, source_chunk = retrieve_corpus_file(
            state["query"], role=state["role"], department=state.get("department"),
        )
        if target:
            log_event(
                state,
                action="Vision Retrieval",
                details=(f"No attachment; retrieved {os.path.basename(target)} "
                         f"from the corpus (RBAC role={state.get('role')})."),
                network_target=QDRANT_URL,
            )

    if not target:
        accepted = ", ".join(sorted(e.lstrip(".") for e in INGESTIBLE_EXTS))
        state["answer"] = (
            "No document matching this request is readable by role "
            f"'{state['role']}', and nothing was attached. Attach a file "
            f"({accepted}) or ask about a document your role can access."
        )
        log_event(state, action="VISION Agent",
                  details="No attachment and no relevant corpus match.",
                  network_target=None)
        return state

    query = state["query"]
    try:
        result = _get_pipeline().run(
            target,
            command="analyze",
            question=query,
            role=state["role"],
            department=state.get("department"),
        )
    except Exception as exc:
        state["answer"] = f"Vision analysis failed: {exc}"
        log_event(state, action="VISION Agent",
                  details=f"Pipeline error: {type(exc).__name__}: {exc}",
                  network_target=None)
        return state

    parts = []
    summary = getattr(result, "summary", None) or getattr(result, "answer", None)
    if summary:
        parts.append(str(summary))

    findings = getattr(result, "key_findings", None) or []
    if findings:
        parts.append("\n**Key findings**\n" + "\n".join(f"- {f}" for f in findings[:10]))

    visuals = getattr(result, "visual_findings", None) or []
    if visuals:
        parts.append("\n**Visual analysis**\n" + "\n".join(f"- {v}" for v in visuals[:5]))

    chunks = getattr(result, "retrieved_chunks", None) or []
    if chunks:
        parts.append("\n**Corpus context**\n" + "\n".join(
            f"- [{c.file_name} | {c.department} | {c.score:.3f}]" for c in chunks[:5]))

    generated = [str(f) for f in (getattr(result, "generated_files", None) or [])]
    if generated:
        parts.append("\n**Generated files**\n" + "\n".join(
            f"- {os.path.basename(f)}" for f in generated))

    # Say which document was analysed when the agent chose it itself -- the
    # user never named it, so an unattributed answer would be unverifiable.
    if source_chunk:
        parts.insert(0, f"**Source:** `{os.path.basename(target)}` "
                        f"({source_chunk.get('department') or 'unknown dept'}), "
                        f"selected from the corpus.\n")

    state["answer"] = "\n".join(parts) if parts else "No analysis produced."
    state["generated_files"] = generated
    state["prompt"] = f"Vision analysis of {os.path.basename(target)}"

    log_event(
        state,
        action="VISION Agent",
        details=(f"Analysed {os.path.basename(target)} "
                 f"({getattr(result, 'detected_type', 'unknown')}): "
                 f"{len(findings)} finding(s), {len(chunks)} corpus chunk(s), "
                 f"{len(generated)} file(s) generated."),
        network_target=get_network_target(),
    )
    return state

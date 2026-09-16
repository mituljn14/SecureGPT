import functools
import time
from retrieval_pipeline.llm.provider import get_network_target
from retrieval_pipeline.graph.logger import log_detailed_session
from retrieval_pipeline.graph.network_classifier import is_local_target

def summarize_sources(chunks, with_ids=False, limit=None):
    """Compact, text-free description of a chunk list for the trace:
    documents with how many chunks each contributed, and (optionally) the
    chunk identities. Identifiers only -- never chunk text, which is what
    would make the trace and the history rows bulky."""
    docs = {}
    for c in (chunks or [])[: limit or None]:
        name = getattr(c, "file_name", None) or getattr(c, "source_path", None) or "unknown"
        d = docs.setdefault(name, {"doc": name, "chunks": 0})
        d["chunks"] += 1
        if with_ids:
            where = None
            if getattr(c, "page", None) is not None:
                where = f"p.{c.page}"
            elif getattr(c, "row_range", None):
                rr = c.row_range
                where = f"row {rr[0]}" if len(rr) < 2 or rr[0] == rr[1] else f"rows {rr[0]}-{rr[1]}"
            d.setdefault("ids", []).append({
                "chunk_id": getattr(c, "chunk_id", None),
                "chunk_index": getattr(c, "chunk_index", None),
                "where": where,
                "score": round(float(getattr(c, "score", 0.0) or 0.0), 3),
            })
    return sorted(docs.values(), key=lambda d: -d["chunks"])


def sources_text(sources):
    return ", ".join(f"{d['doc']}: {d['chunks']}" for d in sources) if sources else "none"


def log_event(state, action, details, network_target=None, sources=None):
    if "trajectory_log" not in state:
        state["trajectory_log"] = []

    # is_local is computed HERE, once, from the real target -- not
    # re-derived later by the frontend via substring-matching the details
    # text (see network_classifier.py's docstring for why that broke).
    event = {
        "action": action,
        "details": details,
        "network_target": network_target,
        "is_local": is_local_target(network_target),
        "timestamp": time.time(),
        # Graph node that emitted this step (set by traced_node), so the
        # trace reads "fast_retrieval: ..." rather than a bare action name.
        "node": state.get("_node"),
    }
    if sources:
        # {"label": [ {doc, chunks, ids?} ... ]} -- which documents fed this step.
        event["sources"] = sources
    state["trajectory_log"].append(event)
    
    # Also log this to the detailed session log
    log_detailed_session("trajectory_event", event)
    
    return state


# Keys worth summarising on node_exit: what the node decided, not the
# whole state (chunks and prompts are large and already logged elsewhere).
_EXIT_KEYS = ("agent_destination", "retrieval_confidence", "crag_score",
              "retrieval_stage", "generated_files")


def traced_node(name, fn):
    """Wrap a LangGraph node so the session log shows the graph itself:
    node_enter / node_exit records with elapsed time and the decisions the
    node wrote to state. The trajectory (user-facing trace) is untouched
    except that its events now carry the emitting node's name."""
    @functools.wraps(fn)
    def wrapped(state):
        state["_node"] = name
        log_detailed_session("node_enter", {
            "node": name, "query": state.get("query"), "role": state.get("role"),
            "username": state.get("username"), "session_id": state.get("session_id"),
        })
        t0 = time.perf_counter()
        try:
            out = fn(state)
        except Exception as exc:
            log_detailed_session("node_error", {
                "node": name, "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        src = out if isinstance(out, dict) else state
        summary = {k: src.get(k) for k in _EXIT_KEYS if src.get(k) not in (None, "", [])}
        if src.get("retrieved_chunks") is not None:
            summary["retrieved_chunks"] = len(src.get("retrieved_chunks") or [])
        log_detailed_session("node_exit", {
            "node": name, "elapsed_ms": int((time.perf_counter() - t0) * 1000), **summary,
        })
        return out
    return wrapped

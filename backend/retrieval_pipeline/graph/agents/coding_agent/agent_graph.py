"""
agent_graph.py
Step 8: Connect Agent

Wires together: Retriever -> document_registry.db -> File Reader ->
llm_agent (Qwen via Ollama) -> sandbox_executor (Docker) -> Verifier -> File Writer

Orchestrated as a LangGraph state graph with a code-fix retry loop.
"""

import os
import re
import sys
from typing import TypedDict, Optional, Any, List

from langgraph.graph import StateGraph, END

from retrieval_pipeline.graph.telemetry import log_event
from retrieval_pipeline.retrieval.config import QDRANT_URL
from retrieval_pipeline.llm.provider import get_network_target

from .verifier import Verifier, RESULT_MARKER
from .file_writer import FileWriter

# ============================================================
# ADAPT SECTION
# ============================================================

from .retriever import Retriever
_retriever = Retriever()

def retrieve_chunks(query: str, role: str, top_k: int = 8,
                    department: Optional[str] = None,
                    metadata_filters: Optional[dict] = None) -> List[dict]:
    """RBAC-filtered hybrid retrieval. `role` is required; a failure here
    returns no chunks rather than falling back to an unfiltered search."""
    try:
        return _retriever.search(query, role=role, top_k=top_k,
                                 department=department,
                                 metadata_filters=metadata_filters)
    except Exception as e:
        print(f"[Warning] Retriever search failed: {e}")
        return []


from .document_registry import DocumentRegistry
_registry = DocumentRegistry()

def get_source_path(document_id: str) -> Optional[str]:
    row = _registry.get_by_document_id(document_id)
    return row.get("source_path") if row else None

def find_document_by_name(file_name: str) -> Optional[dict]:
    return _registry.get_by_file_name(file_name)


from .file_reader import FileReader
_file_reader = FileReader()

def read_file_for_task(source_path: str, file_type: str) -> dict:
    return _file_reader.inspect(source_path, file_type=file_type)


from .llm_agent import generate_code as _llm_generate_code

def generate_code(task: str, context_data: Any = None, previous_error: Optional[str] = None,
                  model_override: Optional[str] = None) -> str:
    """Return generated Python code as a string.

    `model_override` carries the model picked in the UI. Without it the
    dropdown silently had no effect on the coding agent.
    """
    kwargs = dict(user_query=task, file_context=context_data, error_context=previous_error)
    if model_override and model_override != "Auto (Supervisor)":
        # llm_agent talks to Ollama natively, so strip the litellm-style
        # "openai/" prefix the provider layer uses.
        kwargs["model"] = model_override.removeprefix("openai/")
    result = _llm_generate_code(**kwargs)
    return result["code"]


from .sandbox_executor import run_in_sandbox as _sandbox_run_in_sandbox

def _resolve_on_disk(source_path: str) -> str:
    """Turn a v2 payload `source_path` into a real path on this machine.

    v2 stores paths RELATIVE to the corpus root ("Operations/COO/.../x.csv"),
    while the sandbox has to copy an actual file. Absolute paths (uploads,
    legacy registry rows) are passed through untouched.
    """
    from pathlib import Path
    from retrieval_pipeline.retrieval.config import CORPUS_ROOT

    p = Path(source_path)
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = Path(CORPUS_ROOT) / source_path
    return str(candidate if candidate.exists() else p)


def run_in_sandbox(code: str, source_path: Optional[str] = None, llm_source_path: Optional[str] = None,
                    timeout_seconds: float = 30.0) -> dict:
    """Execute code in Docker sandbox, providing mapped input files if any."""
    file_mapping = {}
    if source_path:
        basename = os.path.basename(source_path)
        # Keys are every name the generated code might open the file by;
        # the value is always the real location on this host.
        real_path = _resolve_on_disk(source_path)
        file_mapping = {
            source_path: real_path,
            source_path.replace("\\", "/"): real_path,
            f"/workspace/{basename}": real_path,
            f"/app/input/{basename}": real_path,
            basename: real_path,
        }
        if llm_source_path:
            file_mapping[llm_source_path] = real_path

    result = _sandbox_run_in_sandbox(
        code=code,
        file_mapping=file_mapping,
        timeout_seconds=timeout_seconds,
    )
    result["execution_time"] = result.pop("execution_time_seconds", 0.0)
    return result

# ============================================================
# END ADAPT SECTION
# ============================================================

MAX_RETRIES = 3
_verifier = Verifier(max_execution_time=30.0)
_writer = FileWriter()


class AgentState(TypedDict, total=False):
    user_query: str
    retrieved_chunks: List[dict]
    document_id: Optional[str]
    source_path: Optional[str]
    file_type: Optional[str]
    extracted_data: Any
    llm_source_path: Optional[str]
    generated_code: str
    sandbox_output: dict
    verification: dict
    final_answer: Optional[str]
    error: Optional[str]
    attempt: int
    output_file_requested: Optional[dict]
    output_file_path: Optional[str]
    # Caller identity -- role is what makes retrieval RBAC-correct.
    role: str
    department: Optional[str]
    uploaded_files: List[str]
    model_override: Optional[str]
    # Consumed by coding_agent_node in graph/agents/coding.py
    generated_files: List[str]
    trajectory_log: List[dict]


def is_standalone_query(query: str) -> bool:
    """
    Detect if user query is a direct calculation/algorithm query
    that provides its own numbers and does not target enterprise documents.
    """
    if re.search(r"[\w\-.]+\.(?:xlsx|xls|csv|pdf|docx)", query, re.IGNORECASE):
        return False

    corpus_keywords = [
        "budget", "sales", "report", "revenue", "cost center", "procurement",
        "quality", "quarter", "q1", "q2", "q3", "q4", "variance", "actual_inr",
        "budgeted", "fiscal", "expense", "financial", "corpus", "dataset", "sheet",
        "category", "vendor", "contract", "invoice"
    ]
    query_lower = query.lower()
    for kw in corpus_keywords:
        if kw in query_lower:
            return False

    # Check for direct calculation queries like "average of 10, 20 and 30"
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", query)
    if len(numbers) >= 2 and any(k in query_lower for k in ["average", "sum", "mean", "median", "add", "multiply", "divide", "calculate", "compute"]):
        return True

    return False


def node_retrieve(state: AgentState) -> AgentState:
    query = state["user_query"]

    # 1. If standalone math/algorithm query with inline numbers, skip document retrieval
    if is_standalone_query(query):
        return {**state, "retrieved_chunks": [], "document_id": None}

    # 2. Did the user name a specific file? Resolve it through the v2 corpus,
    #    NOT document_registry.db. That database is a v1 artifact holding
    #    ABSOLUTE paths from the machine that built it (e.g.
    #    "C:\\Users\\mitul jain\\..."), so every hit resolves to a file that
    #    does not exist here and the sandbox dies on FileNotFoundError.
    #    Retrieval is RBAC-filtered, so a named file the caller may not read
    #    correctly finds nothing.
    file_match = re.search(r"[\w\-.]+\.(?:xlsx|xls|csv|pdf|docx)", query, re.IGNORECASE)
    target_filename = file_match.group(0) if file_match else None

    # 3. Semantic retrieval via Qdrant (hybrid dense+sparse, RBAC-filtered)
    chunks = retrieve_chunks(query, role=state["role"], top_k=8,
                             department=state.get("department"))
    if not chunks:
        return {**state, "retrieved_chunks": [], "document_id": None}

    if target_filename:
        named = [c for c in chunks
                 if (c.get("file_name") or "").lower() == target_filename.lower()]
        if named:
            chunks = named

    # 3a. Relevance gate. RRF scores are rank-based, so the top hit always
    # *looks* plausible even when the corpus holds nothing relevant for this
    # role -- something has to rank first. Check the raw dense cosine against
    # the shared floor before spending three sandbox attempts writing code
    # against the wrong file.
    #
    # This is also where an RBAC miss surfaces: when the data lives in a
    # department the caller cannot read, retrieval legitimately returns
    # nothing relevant, and saying so is far better than failing several
    # minutes later with "column does not exist".
    if not _retriever.has_real_match(query, role=state["role"],
                                     department=state.get("department")):
        near = ", ".join(dict.fromkeys(
            (c.get("file_name") or "?") for c in chunks[:3]))
        log_event(
            state,
            action="Coding Retrieval",
            details=(f"No match above the relevance floor for role "
                     f"{state.get('role')}; closest: {near}"),
            network_target=QDRANT_URL,
        )
        return {
            **state,
            "retrieved_chunks": chunks,
            "document_id": None,
            "error": (
                f"No document relevant to this request is readable by role "
                f"'{state['role']}'. Closest available: {near}. If this data "
                f"belongs to another department, your role may not have "
                f"access to it."
            ),
        }

    # 3b. Prefer a tabular source. This agent computes over data, so a
    # spreadsheet the question can actually be answered from beats a prose
    # PDF that merely ranks higher semantically.
    tabular = [c for c in chunks
               if str(c.get("file_name") or "").lower().endswith(
                   (".csv", ".xlsx", ".xls"))]
    ranked = tabular or chunks

    top = sorted(ranked, key=lambda c: c.get("score", 0), reverse=True)[0]
    log_event(
        state,
        action="Coding Retrieval",
        details=(f"Hybrid dense+sparse search (RBAC role={state.get('role')}) "
                 f"matched {len(chunks)} chunk(s); target document: "
                 f"{top.get('file_name')}"),
        network_target=QDRANT_URL,
    )
    # The v2 payload carries source_path directly, so the registry lookup
    # that follows is only a fallback for documents predating it.
    return {
        **state,
        "retrieved_chunks": chunks,
        "document_id": top.get("document_id"),
        "source_path": top.get("source_path"),
    }


def node_lookup_registry(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    # Already resolved from the v2 payload -- skip the registry entirely.
    existing = state.get("source_path")
    if existing:
        ext = os.path.splitext(existing)[1].lstrip(".").lower()
        return {**state, "source_path": existing, "file_type": ext}

    doc_id = state.get("document_id")
    if not doc_id:
        return {**state, "source_path": None, "file_type": None}

    row = _registry.get_by_document_id(doc_id)
    if not row or not row.get("source_path"):
        return {**state, "source_path": None, "file_type": None}

    source_path = row["source_path"]
    ext = os.path.splitext(source_path)[1].lstrip(".").lower()
    return {**state, "source_path": source_path, "file_type": ext}


def node_read_file(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    source_path = state.get("source_path")
    if not source_path:
        return {**state, "extracted_data": None, "llm_source_path": None}

    try:
        data = read_file_for_task(source_path, state.get("file_type", ""))
        return {**state, "extracted_data": data, "llm_source_path": data.get("source_path")}
    except Exception as e:
        print(f"[Warning] FileReader failed: {e}. Falling back to standalone execution.")
        return {**state, "extracted_data": None, "llm_source_path": None}


def node_generate_code(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    prev_error = state.get("sandbox_output", {}).get("stderr") if state.get("attempt", 0) > 0 else None
    try:
        code = generate_code(
            task=state["user_query"],
            context_data=state.get("extracted_data"),
            previous_error=prev_error,
            model_override=state.get("model_override"),
        )
        attempt = state.get("attempt", 0) + 1
        log_event(
            state,
            action="Code Generation",
            details=f"Generated Python for the sandbox (attempt {attempt}/{MAX_RETRIES}).",
            network_target=get_network_target(),
        )
        return {**state, "generated_code": code, "attempt": attempt}
    except Exception as e:
        return {**state, "error": f"LLM code generation failed: {e}"}


def node_run_sandbox(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    output = run_in_sandbox(
        code=state["generated_code"],
        source_path=state.get("source_path"),
        llm_source_path=state.get("llm_source_path"),
    )
    # network_target is deliberately NOT the LLM's: the container runs with
    # --network none, so it is structurally offline regardless of which
    # backend generated the code. Reporting them as one line would make
    # neither independently verifiable.
    log_event(
        state,
        action="Sandbox Execution",
        details=(f"Ran generated code in an isolated Docker container "
                 f"(exit_code={output.get('exit_code')}, "
                 f"{output.get('execution_time', 0.0):.3f}s)."),
        # Exact label from network_classifier._KNOWN_LOCAL_LABELS -- a
        # free-text description would be classified EXTERNAL.
        network_target="Docker Sandbox (network: none)",
    )
    return {**state, "sandbox_output": output}


def node_verify(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    result = _verifier.verify(state["sandbox_output"])
    # verify() returns a VerificationResult dataclass, not a dict.
    _status = getattr(result, "status", None)
    _reason = getattr(result, "reason", None)
    log_event(
        state,
        action="Verification",
        details=(f"Checked sandbox output: status={_status}"
                 + (f" -- {_reason}" if _reason else "")),
        network_target=None,
    )
    return {**state, "verification": result.to_dict()}


def detect_output_file_request(query: str) -> Optional[dict]:
    """
    Detect if user wants to save/export the result to a report file
    (e.g., 'save as csv', 'export to xlsx', 'generate report', 'save to report.csv').
    """
    query_lower = query.lower()

    fmt = None
    if any(k in query_lower for k in ["as csv", "to csv", "format csv", ".csv"]):
        fmt = "csv"
    elif any(k in query_lower for k in ["as xlsx", "to xlsx", "as excel", "to excel", ".xlsx"]):
        fmt = "xlsx"
    elif any(k in query_lower for k in ["as docx", "to docx", "as word", "to word", ".docx"]):
        fmt = "docx"
    elif any(k in query_lower for k in ["as json", "to json", ".json"]):
        fmt = "json"
    elif any(k in query_lower for k in ["generate report", "save report", "export report", "create report", "generate a report", "save as report"]):
        fmt = "xlsx"

    if not fmt:
        return None

    filename = "analysis_report"
    match = re.search(r'(?:as|to|name|filename|named|into)\s+["\']?([A-Za-z0-9_\-]+)(?:\.(?:csv|xlsx|docx|json))?["\']?', query, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        if name.lower() not in ("csv", "xlsx", "docx", "json", "excel", "word", "a", "the", "file", "report"):
            filename = name
        elif "budget" in query_lower:
            filename = "budget_report"
        elif "sales" in query_lower:
            filename = "sales_report"

    return {"format": fmt, "filename": filename}


def node_write_output_file(state: AgentState) -> AgentState:
    req = state.get("output_file_requested")
    if not req or state.get("error") or state.get("verification", {}).get("status") != "pass":
        return state

    fmt = req.get("format", "xlsx")
    filename = req.get("filename", "report")
    answer_data = state["verification"]["result"]
    details = answer_data.get("details", {}) if isinstance(answer_data, dict) else {}

    # Extract or construct structured tabular rows for the report
    rows = []
    if isinstance(details, dict):
        if "grouped_data" in details and isinstance(details["grouped_data"], list):
            rows = details["grouped_data"]
        elif "rows" in details and isinstance(details["rows"], list):
            rows = details["rows"]
        else:
            row = {"Answer": answer_data.get("answer")}
            for k, v in details.items():
                row[str(k).replace("_", " ").title()] = v
            rows = [row]
    else:
        rows = [{"Answer": str(answer_data)}]

    try:
        if fmt == "csv":
            path = _writer.write_csv(filename, rows)
        elif fmt == "xlsx":
            sheets = {"Report": rows}
            path = _writer.write_xlsx(filename, sheets)
        elif fmt == "docx":
            paragraphs = [
                f"Query: {state['user_query']}",
                f"Final Answer: {answer_data.get('answer') if isinstance(answer_data, dict) else answer_data}",
                f"Details:\n{json.dumps(details, indent=2, default=str)}"
            ]
            path = _writer.write_docx(filename, f"Analysis Report: {filename}", paragraphs)
        else:
            path = _writer.write_json(filename, answer_data)
        log_event(
            state,
            action="File Export",
            details=f"Wrote requested artifact: {os.path.basename(str(path))}",
            network_target=None,
        )
        return {**state, "output_file_path": path}
    except Exception as e:
        print(f"[Warning] File writer error: {e}")
        return {**state, "error": f"File writer failed: {e}"}


def _publish_generated_files(state: AgentState) -> list:
    """Surface any written artifact as generated_files, which
    coding_agent_node turns into the chat download link."""
    files = list(state.get("generated_files") or [])
    out = state.get("output_file_path")
    if out and out not in files:
        files.append(out)
    return files


def _fmt_value(v):
    """Numbers with thousands separators and sane precision; strings that
    are really numbers (the sandbox JSON often stringifies numpy ints) too."""
    if isinstance(v, bool):
        return str(v)
    is_num_str = isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit()
    if isinstance(v, (int, float)) or is_num_str:
        n = float(v)
        return f"{int(n):,}" if n.is_integer() else f"{n:,.2f}"
    return str(v)


def _format_answer(answer) -> str:
    """The verified answer as readable markdown: a dict becomes one line per
    key, a scalar stays a single line. Keys are the ones the generated code
    chose, so nothing is invented here -- only presented."""
    if isinstance(answer, dict):
        lines = [f"- **{str(k).replace('_', ' ').capitalize()}:** {_fmt_value(v)}" for k, v in answer.items()]
        return "\n".join(lines) if lines else "No result."
    if isinstance(answer, list):
        return "\n".join(f"- {_fmt_value(x)}" for x in answer)
    return f"**Answer:** {_fmt_value(answer)}"


def node_finalize(state: AgentState) -> AgentState:
    files = _publish_generated_files(state)

    if state.get("error"):
        return {**state, "final_answer": f"Failed: {state['error']}",
                "generated_files": files}

    verification = state.get("verification", {})
    if verification.get("status") == "pass":
        answer = verification.get("result", {}).get("answer")
        msg = _format_answer(answer)
        if state.get("output_file_path"):
            msg += "\n\nSaved as **" + os.path.basename(str(state["output_file_path"])) + "**"
        return {**state, "final_answer": msg, "generated_files": files}

    reason = verification.get("reason", "Verification failed")
    return {**state,
            "final_answer": f"Failed after {state.get('attempt', 0)} attempt(s): {reason}",
            "generated_files": files}


def route_after_verify(state: AgentState) -> str:
    if state.get("error"):
        return "finalize"
    if state.get("verification", {}).get("status") == "pass":
        return "write_output" if state.get("output_file_requested") else "finalize"
    if state.get("attempt", 0) >= MAX_RETRIES:
        return "finalize"
    return "retry_generate"


def route_after_retrieve(state: AgentState) -> str:
    return "finalize" if state.get("error") else "lookup_registry"


def route_after_registry(state: AgentState) -> str:
    return "finalize" if state.get("error") else "read_file"


def route_after_read(state: AgentState) -> str:
    return "finalize" if state.get("error") else "generate_code"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("lookup_registry", node_lookup_registry)
    g.add_node("read_file", node_read_file)
    g.add_node("generate_code", node_generate_code)
    g.add_node("run_sandbox", node_run_sandbox)
    g.add_node("verify", node_verify)
    g.add_node("write_output", node_write_output_file)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"finalize": "finalize", "lookup_registry": "lookup_registry"})
    g.add_conditional_edges("lookup_registry", route_after_registry, {"finalize": "finalize", "read_file": "read_file"})
    g.add_conditional_edges("read_file", route_after_read, {"finalize": "finalize", "generate_code": "generate_code"})
    g.add_edge("generate_code", "run_sandbox")
    g.add_edge("run_sandbox", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"retry_generate": "generate_code", "write_output": "write_output", "finalize": "finalize"})
    g.add_edge("write_output", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run_agent_query(app, query: str, role: str = None, department=None,
                    uploaded_files=None, model_override=None):
    """Run the coding agent.

    `role` is threaded into retrieval so the sandbox can only ever reach
    documents the caller is cleared for. It is keyword-optional purely so
    the __main__ console below still works; the graph node always passes it.
    """
    file_req = detect_output_file_request(query)
    initial_state: AgentState = {
        "user_query": query,
        "attempt": 0,
        "output_file_requested": file_req,
        "role": role,
        "department": department,
        "uploaded_files": uploaded_files or [],
        "model_override": model_override,
        "generated_files": [],
        "trajectory_log": [],
    }
    print(f"\n[Processing Query]: {query!r}")
    if file_req:
        print(f"[Report Export Requested]: format={file_req['format']}, filename={file_req['filename']}")
    print()
    final_state = app.invoke(initial_state)
    print("=" * 60)
    print("FINAL ANSWER:")
    print(final_state.get("final_answer"))
    print("=" * 60)
    print("Attempts used:", final_state.get("attempt"))
    if final_state.get("output_file_path"):
        print("Report Generated At:", final_state["output_file_path"])
    if final_state.get("generated_code"):
        print("\nGenerated Code:\n" + final_state["generated_code"])
    if final_state.get("verification"):
        print("\nVerification:", final_state["verification"])
    return final_state


if __name__ == "__main__":
    app = build_graph()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_agent_query(app, query)
    else:
        print("=" * 60)
        print("SecureGPT Coding Agent Interactive Console")
        print("Type your query below (or 'exit' / 'quit' to stop):")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nEnter your query: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting Coding Agent. Goodbye!")
                    break
                run_agent_query(app, user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Coding Agent. Goodbye!")
                break
import asyncio

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from retrieval_pipeline.api.dependencies import get_current_user
from retrieval_pipeline.api.schemas import ChatRequest, ChatResponse, User
from retrieval_pipeline.graph.workflow import graph
from retrieval_pipeline.graph.logger import LOGS_DIR
from retrieval_pipeline.api import chat_store
import os
import time
import zipfile
import tempfile
import uuid

router = APIRouter(tags=["chat"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.get("/logs/download")
async def download_logs():
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for root, _, files in os.walk(LOGS_DIR):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
    return FileResponse(temp_zip.name, media_type="application/zip", filename="session_logs.zip")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    # Return absolute path so the agent can find it locally
    return {"file_path": os.path.abspath(file_path)}

# Agents write their artifacts to their own output trees, not one shared
# folder: the coding agent writes beside its sandbox, and the vision agent
# splits charts / extracted tables / summaries into separate directories.
# Serving only data/outputs meant every vision chart 404'd.
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_ROOTS = [
    os.path.abspath(OUTPUT_DIR),
    os.path.join(_BACKEND_ROOT, "retrieval_pipeline", "graph", "agents", "coding_agent", "outputs"),
    os.path.join(_BACKEND_ROOT, "vision_agent", "outputs"),
    os.path.join(os.path.dirname(_BACKEND_ROOT), "tmp_minutes"),
]

_INLINE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
}


def resolve_output_file(filename: str):
    """Find a generated artifact by name across the agents' output trees.

    Only the basename is honoured and every candidate is checked to resolve
    *inside* an allowed root, so a crafted name cannot walk out of them.
    """
    safe = os.path.basename(filename)
    if not safe or safe in (".", ".."):
        return None
    for root in OUTPUT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if safe in files:
                candidate = os.path.abspath(os.path.join(dirpath, safe))
                if candidate.startswith(os.path.abspath(root)):
                    return candidate
    return None


@router.get("/download/{filename}")
async def download_generated_file(filename: str):
    file_path = resolve_output_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in _INLINE_TYPES:
        # Charts are rendered inline in the chat, so they must be served as
        # a viewable image rather than as an attachment download.
        return FileResponse(file_path, media_type=_INLINE_TYPES[ext])
    return FileResponse(file_path, filename=os.path.basename(file_path))

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    initial_state = {
        "query": request.query,
        "role": current_user.role,
        "department": current_user.department,
        "trajectory_log": [],
        "uploaded_files": request.uploaded_files or [],
        "generated_files": [],
        "model_override": request.model_override,
        # Lets node_enter / node_exit records in the session log be tied
        # to the chat session they served.
        "session_id": request.session_id,
        "username": current_user.username,
    }
    
    # graph.invoke() is SYNCHRONOUS and can run for minutes (coding-agent
    # queries: retrieval + LLM code-gen + Docker sandbox + verification).
    # Calling it directly here would block this process's single asyncio
    # event loop for that whole duration -- confirmed live: while one
    # coding-agent request was in flight, even a plain GET /docs from a
    # separate curl call returned nothing at all (connection accepted,
    # zero response) until it finished. That means ONE user's long query
    # freezes the entire API for EVERY user, not just that one request.
    # asyncio.to_thread() runs the blocking call in a worker thread instead,
    # so the event loop stays free to serve other requests concurrently.
    #
    # WARNING: On Windows, running PyTorch (which `bge-m3` uses) inside a 
    # ThreadPoolExecutor (which asyncio.to_thread uses) causes an unrecoverable 
    # thread deadlock. We MUST run it in the main thread here to avoid the 5-minute hang.
    _started = time.perf_counter()
    final_state = graph.invoke(initial_state)
    elapsed_ms = int((time.perf_counter() - _started) * 1000)
    
    # Return EVERY artifact, not just the first. A single vision run produces
    # an extracted table, a chart and a summary; picking generated_files[0]
    # surfaced the CSV and silently dropped the chart the user asked for.
    artifacts = []
    for path in final_state.get("generated_files") or []:
        name = os.path.basename(str(path))
        ext = os.path.splitext(name)[1].lower()
        artifacts.append({
            "name": name,
            "url": f"/download/{name}",
            "kind": "image" if ext in _INLINE_TYPES else "file",
        })

    # Prefer an image for the single-download field so the chat can render a
    # chart inline; fall back to the first artifact otherwise.
    primary = next((a for a in artifacts if a["kind"] == "image"), None) or (
        artifacts[0] if artifacts else None)

    answer = final_state.get("answer", "No answer generated")
    trajectory = final_state.get("trajectory_log", [])

    # Persist the turn. Wrapped so a history failure can never lose an answer
    # the agents just spent minutes producing.
    try:
        sid = chat_store.ensure_session(
            current_user.username, current_user.role,
            request.session_id, request.query,
        )
        chat_store.append_message(
            sid, "user", request.query,
            files=[os.path.basename(p) for p in (request.uploaded_files or [])],
        )
        chat_store.append_message(
            sid, "agent", answer,
            artifacts=artifacts, trajectory=trajectory,
            model=request.model_override,
            generated_code=final_state.get("generated_code"),
            structured=final_state.get("structured"),
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:  # pragma: no cover - history is best-effort
        print(f"[chat] history write failed: {type(exc).__name__}: {exc}")
        sid = request.session_id

    return {
        "response": answer,
        "trajectory": trajectory,
        "download_url": primary["url"] if primary else None,
        "artifacts": artifacts,
        "session_id": sid,
        "elapsed_ms": elapsed_ms,
        "generated_code": final_state.get("generated_code"),
        "structured": final_state.get("structured"),
    }

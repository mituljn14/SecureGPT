"""
Meeting Agent API Routes
------------------------
POST /meeting/process          — Upload audio, run the full pipeline, return results
GET  /meeting/download/{job_id} — Stream the generated DOCX file
POST /meeting/{job_id}/ask      — Q&A on a specific meeting transcript
POST /meeting/query             — Unified Meeting Intelligence Q&A endpoint
GET  /meeting/history           — List all processed meetings in the system

All endpoints are JWT-protected (same Depends(get_current_user) as /chat).
"""
import os
import uuid
import tempfile
import datetime
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from retrieval_pipeline.api.dependencies import get_current_user
from retrieval_pipeline.api.schemas import (
    User,
    MeetingProcessResponse,
    MeetingQARequest,
    MeetingQAResponse,
    MeetingHistoryItem,
    MeetingHistoryResponse,
)
from retrieval_pipeline.llm.provider import generate
from meeting_agent.workflow import meeting_graph
from meeting_agent.query_engine import answer_meeting_question

logger = logging.getLogger(__name__)

router = APIRouter(tags=["meeting"])

# In-memory job store: {job_id: MeetingState}
_job_store: Dict[str, Dict[str, Any]] = {}

# Accepted audio MIME types / extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}


def _validate_audio(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: '{ext}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


# ---------------------------------------------------------------------------
# POST /meeting/process
# ---------------------------------------------------------------------------

@router.post("/process", response_model=MeetingProcessResponse)
async def process_meeting(
    audio: UploadFile = File(..., description="Meeting recording (wav/mp3/m4a/flac)"),
    meeting_title: str = Form(default="Internal Meeting"),
    enable_diarization: bool = Form(default=True),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a meeting recording. The pipeline:
    1. Transcribes with local Whisper
    2. Optionally diarizes speakers
    3. Runs the MeetingAgent LangGraph (summary / decisions / action items)
    4. Builds a DOCX meeting minutes file
    5. Indexes transcript into local/remote Qdrant
    6. Returns the structured results + a download link
    """
    _validate_audio(audio.filename or "recording.wav")

    # ── Save upload to temp file ────────────────────────────────────────────
    suffix = Path(audio.filename or "recording.wav").suffix.lower()
    job_id = str(uuid.uuid4())

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix=f"meeting_{job_id}_"
        ) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {exc}")

    logger.info(
        f"[Meeting/process] job_id={job_id}, file={audio.filename}, "
        f"user={current_user.username}, size={len(content)} bytes"
    )

    # ── Build initial state ─────────────────────────────────────────────────
    initial_state = {
        "job_id": job_id,
        "audio_path": tmp_path,
        "enable_diarization": enable_diarization,
        "meeting_title": meeting_title,
        "user_role": current_user.role,
        "user_department": current_user.department,
        # Outputs (will be populated by nodes)
        "raw_transcript": "",
        "diarized_transcript": "",
        "meeting_summary": "",
        "decisions": [],
        "action_items": [],
        "rag_context": "",
        "minutes_path": "",
        "indexed": False,
        "indexing_error": None,
        "error": None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Run LangGraph pipeline ──────────────────────────────────────────────
    try:
        final_state = meeting_graph.invoke(initial_state)
    except Exception as exc:
        logger.error(f"[Meeting/process] Pipeline error for job {job_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")
    finally:
        # Clean up temp audio file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if final_state.get("error"):
        logger.warning(f"[Meeting/process] Node error in job {job_id}: {final_state['error']}")

    # ── Store result for download & QA endpoints ────────────────────────────
    _job_store[job_id] = final_state

    minutes_available = bool(
        final_state.get("minutes_path") and
        Path(final_state["minutes_path"]).exists()
    )

    return MeetingProcessResponse(
        job_id=job_id,
        meeting_title=meeting_title,
        summary=final_state.get("meeting_summary", ""),
        decisions=final_state.get("decisions", []),
        action_items=final_state.get("action_items", []),
        transcript=final_state.get("diarized_transcript") or final_state.get("raw_transcript", ""),
        minutes_available=minutes_available,
        indexed=final_state.get("indexed", False),
        indexing_error=final_state.get("indexing_error"),
        error=final_state.get("error"),
    )


# ---------------------------------------------------------------------------
# GET /meeting/download/{job_id}
# ---------------------------------------------------------------------------

@router.get("/download/{job_id}")
async def download_minutes(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Download the generated DOCX meeting minutes for a completed job.
    """
    state = _job_store.get(job_id)
    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired or not been processed yet."
        )

    minutes_path = state.get("minutes_path", "")
    if not minutes_path or not Path(minutes_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Minutes file not found. The pipeline may have failed."
        )

    filename = f"meeting_minutes_{job_id[:8]}.docx"
    return FileResponse(
        path=minutes_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# GET /meeting/history
# ---------------------------------------------------------------------------

@router.get("/history", response_model=MeetingHistoryResponse)
async def get_meeting_history(
    current_user: User = Depends(get_current_user),
):
    """
    Returns list of all processed meetings available in this session.
    """
    items = []
    for j_id, st in reversed(list(_job_store.items())):
        items.append(
            MeetingHistoryItem(
                job_id=j_id,
                meeting_title=st.get("meeting_title", "Internal Meeting"),
                summary=st.get("meeting_summary", "")[:200] + ("..." if len(st.get("meeting_summary", "")) > 200 else ""),
                decisions_count=len(st.get("decisions", [])),
                action_items_count=len(st.get("action_items", [])),
                created_at=st.get("created_at", "Just now"),
                minutes_available=bool(st.get("minutes_path") and Path(st.get("minutes_path", "")).exists()),
                indexed=st.get("indexed", False),
            )
        )
    return MeetingHistoryResponse(meetings=items)


# ---------------------------------------------------------------------------
# POST /meeting/{job_id}/ask
# ---------------------------------------------------------------------------

@router.post("/{job_id}/ask", response_model=MeetingQAResponse)
async def ask_meeting_question_endpoint(
    job_id: str,
    payload: MeetingQARequest,
    current_user: User = Depends(get_current_user),
):
    """
    Ask a natural-language question about a processed meeting transcript.
    Grounds answer on transcript chunks retrieved from Qdrant or in-memory state.
    """
    actual_job_id = job_id
    question = payload.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Look up in-memory transcript & title
    st = _job_store.get(actual_job_id, {})
    in_mem_transcript = (
        payload.custom_transcript or
        st.get("diarized_transcript") or
        st.get("raw_transcript") or
        ""
    )
    m_title = payload.meeting_title or st.get("meeting_title", "Meeting Recording")

    try:
        result = answer_meeting_question(
            question=question,
            job_id=actual_job_id,
            user_role=current_user.role,
            user_department=current_user.department,
            in_memory_transcript=in_mem_transcript,
            meeting_title=m_title,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}")

    return MeetingQAResponse(
        job_id=actual_job_id,
        question=question,
        answer=result.get("answer", ""),
        sources=result.get("sources", [])
    )


# ---------------------------------------------------------------------------
# POST /meeting/query
# ---------------------------------------------------------------------------

@router.post("/query", response_model=MeetingQAResponse)
async def query_meetings_general_endpoint(
    payload: MeetingQARequest,
    current_user: User = Depends(get_current_user),
):
    """
    Unified Meeting Intelligence Query endpoint.
    Can query across all indexed meetings, or a specific meeting if job_id is provided.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    target_job_id = payload.job_id or ""
    st = _job_store.get(target_job_id, {}) if target_job_id else {}

    # If no job_id specified but there's a recent meeting in _job_store, use it as context fallback
    in_mem_transcript = payload.custom_transcript or st.get("diarized_transcript") or st.get("raw_transcript")
    if not in_mem_transcript and _job_store:
        # Take latest meeting transcript
        latest_job = list(_job_store.values())[-1]
        in_mem_transcript = latest_job.get("diarized_transcript") or latest_job.get("raw_transcript")
        m_title = payload.meeting_title or latest_job.get("meeting_title")
    else:
        m_title = payload.meeting_title or st.get("meeting_title")

    try:
        result = answer_meeting_question(
            question=question,
            job_id=target_job_id,
            user_role=current_user.role,
            user_department=current_user.department,
            in_memory_transcript=in_mem_transcript,
            meeting_title=m_title,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}")

    return MeetingQAResponse(
        job_id=target_job_id or "all",
        question=question,
        answer=result.get("answer", ""),
        sources=result.get("sources", [])
    )

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None

class User(BaseModel):
    username: str
    role: str
    department: str

class ChatRequest(BaseModel):
    query: str
    uploaded_files: Optional[List[str]] = None
    model_override: Optional[str] = None
    # Omit to start a new session; the server titles it from this query.
    session_id: Optional[str] = None

class Artifact(BaseModel):
    name: str
    url: str
    kind: str  # "image" renders inline in the chat; "file" is a download


class ChatResponse(BaseModel):
    response: str
    trajectory: list[dict] = []
    # Kept for compatibility. Points at the image when the turn produced one,
    # so a chart can be rendered inline rather than only downloaded.
    download_url: Optional[str] = None
    artifacts: List[Artifact] = []
    # Session this turn was stored under; echoed back so a client that sent
    # no session_id can keep using the one the server created.
    session_id: Optional[str] = None
    # Wall-clock for the whole graph run, so the UI reports a measured
    # duration rather than a decorative one.
    elapsed_ms: Optional[int] = None
    # Python the coding agent wrote, for the Code tab.
    generated_code: Optional[str] = None
    # Verified {answer, details} computed in the sandbox.
    structured: Optional[dict] = None


# ---------------------------------------------------------------------------
# Meeting Agent schemas -- brought in with the meeting agent.
# ---------------------------------------------------------------------------

class ActionItem(BaseModel):
    task: str
    owner: str
    deadline: str
    priority: str   # High | Medium | Low


class Decision(BaseModel):
    decision: str
    rationale: str


class MeetingProcessResponse(BaseModel):
    job_id: str
    meeting_title: str
    summary: str
    decisions: List[Decision]
    action_items: List[ActionItem]
    transcript: str
    minutes_available: bool
    indexed: bool = False
    indexing_error: Optional[str] = None
    error: Optional[str] = None


class MeetingQARequest(BaseModel):
    job_id: Optional[str] = None
    question: str
    custom_transcript: Optional[str] = None
    meeting_title: Optional[str] = None


class MeetingQAResponse(BaseModel):
    job_id: str
    question: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class MeetingHistoryItem(BaseModel):
    job_id: str
    meeting_title: str
    summary: str
    decisions_count: int = 0
    action_items_count: int = 0
    created_at: str
    minutes_available: bool
    indexed: bool


class MeetingHistoryResponse(BaseModel):
    meetings: List[MeetingHistoryItem]

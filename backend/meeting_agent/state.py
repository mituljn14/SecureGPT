"""
MeetingState — LangGraph state for the Meeting Audio Agent pipeline.

Flows through: STT → Diarization → Agent (summarize/decisions/actions/RAG) → MinutesBuilder
"""
from typing import TypedDict, List, Optional


class ActionItem(TypedDict):
    task: str
    owner: str       # "Unknown" if not stated
    deadline: str    # e.g., "24 hours", "2026-09-01", "Not specified"
    priority: str    # "High" | "Medium" | "Low"


class Decision(TypedDict):
    decision: str
    rationale: str   # empty string if not stated


class MeetingState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    job_id: str
    audio_path: str            # absolute path to the uploaded temp file
    enable_diarization: bool
    meeting_title: str         # optional title supplied by the user
    user_role: str             # from JWT
    user_department: str       # from JWT

    # ── STT ──────────────────────────────────────────────────
    raw_transcript: str        # plain text, no speaker labels

    # ── Diarization ──────────────────────────────────────────
    diarized_transcript: str   # text annotated with [SPEAKER_X]: …

    # ── Agent outputs ────────────────────────────────────────
    meeting_summary: str
    decisions: List[Decision]
    action_items: List[ActionItem]
    rag_context: str           # extra enterprise context from Qdrant (may be "")

    # ── Output ───────────────────────────────────────────────
    minutes_path: str          # absolute path to the generated .docx
    indexed: bool              # whether the transcript was indexed into Qdrant
    indexing_error: Optional[str] # error message if indexing failed
    error: Optional[str]       # set on failure


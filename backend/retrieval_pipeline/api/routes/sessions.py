"""
Chat session history endpoints.

Every route is JWT-protected and scoped to the caller's username, so one
user's history is never reachable from another's token. A session that does
not belong to the caller returns 404 rather than 403 -- not confirming that
someone else's session id exists.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from retrieval_pipeline.api import chat_store
from retrieval_pipeline.api.dependencies import get_current_user
from retrieval_pipeline.api.schemas import User

router = APIRouter(tags=["sessions"])


class NewSession(BaseModel):
    title: str = "New session"


class RenameSession(BaseModel):
    title: str


@router.get("/sessions")
async def list_sessions(current_user: User = Depends(get_current_user)):
    return {"sessions": chat_store.list_sessions(current_user.username)}


@router.post("/sessions")
async def create_session(body: NewSession, current_user: User = Depends(get_current_user)):
    return chat_store.create_session(current_user.username, current_user.role, body.title)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, current_user: User = Depends(get_current_user)):
    messages = chat_store.get_messages(current_user.username, session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": session_id, "messages": messages}


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, body: RenameSession,
                         current_user: User = Depends(get_current_user)):
    if not chat_store.rename_session(current_user.username, session_id, body.title):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    if not chat_store.delete_session(current_user.username, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.get("/audit")
async def audit_stream(limit: int = 100, current_user: User = Depends(get_current_user)):
    """Last `limit` trace events for the signed-in user, across all their
    chats -- what the Security page's audit stream shows. Capped at 500."""
    return chat_store.recent_events(current_user.username, max(1, min(limit, 500)))

import sys

# Force UTF-8 on the console BEFORE anything else imports and starts
# printing. On Windows, sys.stdout defaults to the cp1252 code page, which
# has no mapping for the rupee sign -- so a single print() of any answer,
# query, or generated line containing "₹" raises UnicodeEncodeError and
# takes the whole request down with it. This corpus is Indian financial
# data: rupee signs are everywhere, and the agent prints the user query,
# the generated code, and the final answer on every run. errors="replace"
# means an unmappable character degrades to a placeholder instead of
# killing a live demo.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # already-wrapped or non-reconfigurable stream (e.g. under pytest)

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from retrieval_pipeline.api.routes import auth, chat

app = FastAPI(title="SecureGPT Enterprise Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API audit: one session-log record per request. Alongside the node_enter /
# node_exit and trajectory records the graph writes, logs/session.jsonl then
# reads as a complete account of a turn: which endpoint was hit, by whom,
# how long it took, and every node it passed through.
# ---------------------------------------------------------------------------
import time as _time

from starlette.requests import Request as _Request

from retrieval_pipeline.graph.logger import log_detailed_session as _log_session


def _identity(request: _Request):
    """Username/role from the bearer token, without failing the request if
    the token is missing or bad -- auth is enforced by the route, not here."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None, None
    try:
        from jose import jwt as _jwt
        from retrieval_pipeline.api.security import SECRET_KEY, ALGORITHM
        claims = _jwt.decode(auth_header[7:], SECRET_KEY, algorithms=[ALGORITHM])
        return claims.get("sub"), claims.get("role")
    except Exception:
        return None, None


@app.middleware("http")
async def _audit_requests(request: _Request, call_next):
    if request.method == "OPTIONS":            # CORS preflight: no audit value
        return await call_next(request)
    t0 = _time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        username, role = _identity(request)
        _log_session("api_request", {
            "method": request.method,
            "path": request.url.path,
            "query_string": request.url.query or None,
            "status": status,
            "elapsed_ms": int((_time.perf_counter() - t0) * 1000),
            "username": username,
            "role": role,
            "client": request.client.host if request.client else None,
        })


app.include_router(auth.router)
app.include_router(chat.router)

from retrieval_pipeline.api.routes import stats, sessions, search  # noqa: E402
app.include_router(stats.router)
app.include_router(sessions.router)
app.include_router(search.router)

# The meeting agent pulls in heavy optional dependencies (openai-whisper,
# and pyannote.audio if enabled). Registering it conditionally means a
# machine without those installed still gets a working /chat rather than a
# server that refuses to boot.
try:
    from retrieval_pipeline.api.routes import meeting
    app.include_router(meeting.router, prefix="/meeting")
    _MEETING_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - depends on local install
    logging.getLogger(__name__).warning(
        "Meeting agent routes not loaded (%s: %s). Install openai-whisper to "
        "enable /meeting endpoints.", type(_exc).__name__, _exc
    )
    _MEETING_AVAILABLE = False


@app.get("/")
def read_root():
    return {
        "message": "SecureGPT API is running.",
        "agents": {
            "enterprise": True,
            "coding": True,
            "meeting": _MEETING_AVAILABLE,
        },
    }

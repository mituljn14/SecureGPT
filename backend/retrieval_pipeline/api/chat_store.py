"""
Chat history persistence (SQLite).

Sessions and messages were previously held in React state only, so every
refresh or role switch lost the conversation, and nothing survived a backend
restart. This stores them on disk instead.

SQLite via the stdlib rather than an ORM: the schema is two tables, the
workload is a handful of writes per query, and adding SQLAlchemy would be a
dependency for no benefit at this size.

Rows are scoped by `username` from the JWT. That is ownership, not access
control for the corpus -- document-level RBAC stays where it belongs, in the
Qdrant filter.
"""
import json
import os
import sqlite3
import time
import uuid
from typing import List, Optional

_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(_DB_DIR, exist_ok=True)
DB_PATH = os.path.abspath(os.path.join(_DB_DIR, "chat_history.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL,
    role        TEXT,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- 'user' | 'agent'
    text        TEXT NOT NULL,
    files       TEXT,                   -- JSON: attachment names (user turns)
    artifacts   TEXT,                   -- JSON: generated artifacts (agent turns)
    trajectory  TEXT,                   -- JSON: per-step audit trail
    model       TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after the first release. Applied idempotently on start-up so
# an existing chat_history.db keeps every row and simply gains the fields.
_MIGRATIONS = [
    ("messages", "generated_code", "TEXT"),   # Python the coding agent wrote
    ("messages", "structured", "TEXT"),       # JSON: verified {answer, details}
    ("messages", "elapsed_ms", "INTEGER"),    # wall-clock of the graph run
]


def init_db():
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        have = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
        for table, column, ctype in _MIGRATIONS:
            if column not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")


init_db()


def _loads(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def create_session(username: str, role: str, title: str) -> dict:
    sid = uuid.uuid4().hex[:16]
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, username, role, title, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (sid, username, role, title[:120] or "New session", now, now),
        )
    return {"id": sid, "title": title[:120] or "New session", "role": role,
            "created_at": now, "updated_at": now, "message_count": 0}


def list_sessions(username: str, limit: int = 50) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)"
            " AS message_count FROM sessions s WHERE s.username = ?"
            " ORDER BY s.updated_at DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_events(username: str, limit: int = 100) -> dict:
    """The newest `limit` trace events across every chat this user owns,
    newest first -- the per-user audit stream. Each event is the exact
    record the graph wrote (action, details, network_target, is_local,
    node, timestamp), annotated with the session it came from. Only this
    user's sessions are read, so one person's audit never shows another's."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT m.id, m.trajectory, m.session_id, s.title, s.role"
            " FROM messages m JOIN sessions s ON s.id = m.session_id"
            " WHERE s.username = ? AND m.role = 'agent'"
            " ORDER BY m.id DESC LIMIT 400",
            (username,),
        ).fetchall()

    events, sessions = [], set()
    for r in rows:
        trail = _loads(r["trajectory"], [])
        for ev in reversed(trail):          # newest step first within a turn
            if not isinstance(ev, dict):
                continue
            events.append({**ev, "session_id": r["session_id"],
                           "session_title": r["title"], "role": r["role"]})
            sessions.add(r["session_id"])
            if len(events) >= limit:
                break
        if len(events) >= limit:
            break
    return {"events": events, "sessions": len(sessions), "limit": limit}


def get_messages(username: str, session_id: str) -> Optional[List[dict]]:
    """None when the session does not exist or belongs to someone else --
    the caller turns that into a 404 rather than leaking its existence."""
    with _connect() as conn:
        owner = conn.execute(
            "SELECT username FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not owner or owner["username"] != username:
            return None
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()

    out = []
    for r in rows:
        out.append({
            "role": r["role"],
            "text": r["text"],
            "files": _loads(r["files"], []),
            "artifacts": _loads(r["artifacts"], []),
            "trajectory": _loads(r["trajectory"], []),
            "model": r["model"],
            "created_at": r["created_at"],
            # Same keys the live /chat response uses, so a reopened chat
            # renders identically: Code button, metric cards, elapsed time.
            "code": r["generated_code"],
            "structured": _loads(r["structured"], None),
            "elapsedMs": r["elapsed_ms"],
        })
    return out


def append_message(session_id: str, role: str, text: str, *, files=None,
                   artifacts=None, trajectory=None, model=None,
                   generated_code=None, structured=None, elapsed_ms=None) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, text, files, artifacts,"
            " trajectory, model, created_at, generated_code, structured, elapsed_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, role, text,
             json.dumps(files or []), json.dumps(artifacts or []),
             json.dumps(trajectory or []), model, now,
             generated_code, json.dumps(structured) if structured is not None else None,
             elapsed_ms),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))


def rename_session(username: str, session_id: str, title: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ? AND username = ?",
            (title[:120], session_id, username),
        )
        return cur.rowcount > 0


def delete_session(username: str, session_id: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id IN"
                     " (SELECT id FROM sessions WHERE id = ? AND username = ?)",
                     (session_id, username))
        cur = conn.execute("DELETE FROM sessions WHERE id = ? AND username = ?",
                           (session_id, username))
        return cur.rowcount > 0


def ensure_session(username: str, role: str, session_id: Optional[str],
                   first_query: str) -> str:
    """Return a usable session id, creating one titled from the first query
    when the client did not supply a valid one."""
    if session_id:
        with _connect() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ? AND username = ?",
                (session_id, username),
            ).fetchone()
        if row:
            return session_id
    return create_session(username, role, (first_query or "New session").strip())["id"]

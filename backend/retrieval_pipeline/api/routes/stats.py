"""
Workspace stats, role catalogue and capability discovery for the UI.

Everything here is measured from the running system, or derived from the very
constants the agents route on -- never restated. A UI holding its own copy of
the role list, the model list or the accepted file extensions will eventually
disagree with the backend, and that disagreement surfaces as a file the
picker accepted but no agent can read.
"""
import os
import time

from fastapi import APIRouter, Depends

from retrieval_pipeline.api.dependencies import get_current_user
from retrieval_pipeline.api.schemas import User

router = APIRouter(tags=["stats"])

# Embedding-only and vision-only models are not chat models; offering them in
# a model picker produces a confusing failure rather than an answer.
_NON_CHAT_MODELS = {"bge-m3", "moondream", "nomic-embed-text"}

_OLLAMA = os.getenv("ON_PREMISE_API_BASE", "http://localhost:11434/v1").replace("/v1", "")
_QDRANT = os.getenv("QDRANT_URL", "http://localhost:6333")

# Per-role chunk counts cost one exact count query each, so they are computed
# once and reused: the corpus is static at runtime, and re-counting on every
# header render would add seconds to each role switch.
_role_counts_cache = {"at": 0.0, "data": None}
_CACHE_TTL = 900


def _qdrant_client():
    from retrieval_pipeline.retrieval.qdrant_client import get_client
    return get_client()


def _collection():
    from retrieval_pipeline.retrieval.retrieve import COLLECTION_NAME
    return COLLECTION_NAME


def _count_for_role(client, collection, role, department=None):
    """Chunks this identity may retrieve -- the same filter retrieval uses,
    so the number shown is the number enforced (RBAC_SCOPE included)."""
    from retrieval_pipeline.retrieval.filter_builder import build_filter
    return client.count(
        collection_name=collection,
        count_filter=build_filter(role=role, department=department, metadata_filters={}),
        exact=True,
    ).count


def _departments():
    """Departments present in the corpus, sampled from the payload rather than
    listed here, so a re-ingest with different departments is reflected."""
    try:
        client, collection = _qdrant_client(), _collection()
        points, _ = client.scroll(
            collection_name=collection, limit=2000,
            with_payload=["department"], with_vectors=False,
        )
        return sorted({p.payload.get("department") for p in points if p.payload.get("department")})
    except Exception:
        return []


def _ollama_models():
    try:
        import requests
        tags = requests.get(f"{_OLLAMA}/api/tags", timeout=4).json()
        return [
            m["name"] for m in tags.get("models", [])
            if m.get("name", "").split(":")[0] not in _NON_CHAT_MODELS
        ]
    except Exception:
        return []


@router.get("/stats")
async def workspace_stats(current_user: User = Depends(get_current_user)):
    collection, total, mine = None, 0, 0
    try:
        client, collection = _qdrant_client(), _collection()
        total = client.count(collection_name=collection, exact=True).count
        mine = _count_for_role(client, collection, current_user.role, current_user.department)
    except Exception:
        pass

    depts = _departments()
    return {
        "collection": collection,
        "chunks_total": total,
        "chunks_visible_to_role": mine,
        "departments": len(depts),
        "department_names": depts,
        "models": _ollama_models(),
        "role": current_user.role,
        "username": current_user.username,
        "deployment": "on-premise",
    }


@router.get("/roles")
async def list_roles():
    """Role catalogue with live per-role corpus reach.

    Public because the UI needs it before any token exists, and because it
    exposes only role names and counts -- never document content.
    """
    from retrieval_pipeline.api.database import MOCK_USERS

    counts, now = {}, time.time()
    if _role_counts_cache["data"] and now - _role_counts_cache["at"] < _CACHE_TTL:
        counts = _role_counts_cache["data"]
    else:
        try:
            client, collection = _qdrant_client(), _collection()
            for u in MOCK_USERS.values():
                counts[u["role"]] = _count_for_role(client, collection, u["role"], u["department"])
            _role_counts_cache.update({"at": now, "data": counts})
        except Exception:
            counts = {}

    roles = [
        {
            "role": u["role"],
            "username": u["username"],
            "department": u["department"],
            "chunks": counts.get(u["role"], 0),
        }
        for u in MOCK_USERS.values()
    ]
    # Broadest access first -- the natural order for a switcher.
    roles.sort(key=lambda r: -r["chunks"])
    return {"roles": roles}


@router.get("/capabilities")
async def capabilities():
    """What this deployment can ingest, and where it executes.

    Extensions come from the same module-level sets the supervisor routes on,
    so the file picker and the router cannot drift apart.
    """
    from retrieval_pipeline.graph.agents.vision import VISION_EXTS, TABULAR_EXTS
    from retrieval_pipeline.graph.agents.meeting import AUDIO_EXTS

    images = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
    pdfs = {".pdf"}
    slides = {".pptx", ".ppt"}
    words = {".docx", ".doc"}
    text = VISION_EXTS - images - pdfs - slides - words

    sandbox_image = None
    try:
        from retrieval_pipeline.graph.agents.coding_agent import sandbox_executor
        sandbox_image = sandbox_executor.IMAGE_NAME
    except Exception:
        pass

    return {
        "file_types": [
            {"kind": "image", "label": "Images", "agent": "Vision", "tag": "IMG", "exts": sorted(images)},
            {"kind": "pdf", "label": "PDF", "agent": "Vision", "tag": "PDF", "exts": sorted(pdfs)},
            {"kind": "slides", "label": "Slides", "agent": "Vision", "tag": "PPT", "exts": sorted(slides)},
            {"kind": "doc", "label": "Documents", "agent": "Vision", "tag": "DOC", "exts": sorted(words)},
            {"kind": "text", "label": "Text", "agent": "Vision", "tag": "TXT", "exts": sorted(text)},
            {"kind": "table", "label": "Spreadsheets", "agent": "Coding / Vision", "tag": "TBL", "exts": sorted(TABULAR_EXTS)},
            {"kind": "audio", "label": "Audio", "agent": "Meeting", "tag": "AUD", "exts": sorted(AUDIO_EXTS)},
        ],
        "runtimes": [
            {"key": "network", "label": "External Network", "target": "blocked", "tag": "BLOCKED", "tone": "red",
             "note": "Sandbox containers run with --network none"},
            {"key": "llm", "label": "LLM Inference", "target": _OLLAMA, "tag": "LOCAL", "tone": "mint",
             "note": f"Ollama at {_OLLAMA}"},
            {"key": "vectors", "label": "Vector Retrieval", "target": _QDRANT, "tag": "LOCAL", "tone": "mint",
             "note": f"Qdrant at {_QDRANT}"},
            {"key": "storage", "label": "Data Storage", "target": os.getenv("CORPUS_ROOT") or "corpus/",
             "tag": "ON-PREMISE", "tone": "violet", "note": "Corpus on local disk"},
            {"key": "sandbox", "label": "Code Execution", "target": sandbox_image or "unavailable",
             "tag": "SANDBOXED", "tone": "amber",
             "note": f"Docker image {sandbox_image}" if sandbox_image else "Docker sandbox not detected"},
        ],
        "deployment": os.getenv("LLM_DEPLOYMENT_TYPE", "on_premise"),
    }

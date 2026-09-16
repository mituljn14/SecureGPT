"""
Transcript Indexer Node
-----------------------
Chunks the diarized transcript and indexes it into Qdrant for later retrieval.
"""
import os
import uuid
import datetime
import logging
from typing import List, Dict

from qdrant_client.http import models as qmodels
from retrieval_pipeline.retrieval.qdrant_client import get_client
from retrieval_pipeline.retrieval.embedding import embed_texts

logger = logging.getLogger(__name__)

MEETING_COLLECTION_NAME = os.getenv("MEETING_COLLECTION_NAME", "meeting_transcripts")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

def _ensure_collection(client):
    try:
        if client.collection_exists(MEETING_COLLECTION_NAME):
            return

        logger.info(f"[TranscriptIndexer] Creating collection '{MEETING_COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=MEETING_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=EMBED_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )

        # Payload indexes for filtering
        for field in ["job_id", "user_role", "user_department", "source"]:
            try:
                client.create_payload_index(
                    collection_name=MEETING_COLLECTION_NAME,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"[TranscriptIndexer] Collection setup warning: {exc}")

def _chunk_transcript(transcript: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Splits transcript into overlapping chunks by words.
    """
    words = transcript.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def index_transcript_node(state: dict) -> dict:
    """
    LangGraph node: index the diarized_transcript into Qdrant.
    """
    transcript = state.get("diarized_transcript") or state.get("raw_transcript")
    if not transcript:
        state["indexed"] = False
        state["indexing_error"] = "No transcript available to index."
        return state

    job_id = state.get("job_id", "unknown")
    meeting_title = state.get("meeting_title", "Meeting")
    user_role = state.get("user_role", "viewer")
    user_department = state.get("user_department", "")

    try:
        client = get_client()
        _ensure_collection(client)

        chunks = _chunk_transcript(transcript, chunk_size=400, overlap=50)
        
        # Batch embed for performance
        vectors = embed_texts(chunks)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "job_id": job_id,
                "meeting_title": meeting_title,
                "chunk_index": i,
                "user_role": user_role,
                "user_department": user_department,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": "meeting_transcript",
                "text": chunk
            }
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=MEETING_COLLECTION_NAME,
            points=points,
            wait=True,
        )

        state["indexed"] = True
        state["indexing_error"] = None
        logger.info(f"[TranscriptIndexer] Successfully indexed {len(points)} chunks for job {job_id}.")

    except Exception as exc:
        logger.warning(f"[TranscriptIndexer] Indexing failed for job {job_id}: {exc}")
        state["indexed"] = False
        state["indexing_error"] = str(exc)

    return state

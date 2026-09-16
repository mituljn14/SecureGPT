"""
Meeting Query Engine
--------------------
Provides comprehensive Q&A functionality against meeting transcripts:
- Vector retrieval from Qdrant when indexed
- Direct transcript grounding & fallback
- Cross-meeting / general meeting queries
"""
import os
import re
import logging
from typing import Dict, Any, List, Optional

from qdrant_client.http import models as qmodels
from retrieval_pipeline.retrieval.qdrant_client import get_client
from retrieval_pipeline.retrieval.embedding import embed_text
from retrieval_pipeline.llm.provider import generate

logger = logging.getLogger(__name__)

MEETING_COLLECTION_NAME = os.getenv("MEETING_COLLECTION_NAME", "meeting_transcripts")
TOP_K = 5

def _extract_heuristic_sources(transcript: str, question: str, top_n: int = 4) -> List[Dict[str, Any]]:
    """Extract relevant sentence/paragraph windows from transcript based on keyword overlap."""
    if not transcript:
        return []
    
    # Split by lines or double newlines
    paragraphs = [p.strip() for p in re.split(r'\n+', transcript) if len(p.strip()) > 20]
    if not paragraphs:
        paragraphs = [transcript[i:i+400] for i in range(0, len(transcript), 350)]

    q_words = set(re.findall(r'\w+', question.lower())) - {"what", "who", "when", "where", "why", "how", "the", "a", "is", "was", "and", "in", "to", "of", "for", "with", "this", "that"}
    
    scored = []
    for idx, p in enumerate(paragraphs):
        p_words = set(re.findall(r'\w+', p.lower()))
        overlap = len(q_words & p_words)
        scored.append((overlap, idx, p))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:top_n]
    # Re-sort by original index for chronological order
    selected.sort(key=lambda x: x[1])

    return [{"text": s[2], "chunk_index": s[1]} for s in selected]


def answer_meeting_question(
    question: str,
    job_id: Optional[str] = None,
    user_role: str = "viewer",
    user_department: str = "",
    in_memory_transcript: Optional[str] = None,
    meeting_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Answers a question based on meeting transcripts (vector-grounded + transcript-grounded).
    """
    question = question.strip()
    logger.info(f"[QueryEngine] Answering question (job_id={job_id}): {question}")

    sources: List[Dict[str, Any]] = []
    context_parts: List[str] = []

    # 1. Try Qdrant vector retrieval first
    try:
        query_vector = embed_text(question)
        client = get_client()

        if client.collection_exists(MEETING_COLLECTION_NAME):
            must_conditions = []
            if job_id and job_id.lower() not in ("all", "global", "latest", ""):
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="job_id",
                        match=qmodels.MatchValue(value=job_id),
                    )
                )

            query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

            response = client.query_points(
                collection_name=MEETING_COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=TOP_K,
                with_payload=True,
                with_vectors=False,
            )

            for point in response.points:
                p_text = point.payload.get("text", "")
                p_idx = point.payload.get("chunk_index", -1)
                p_title = point.payload.get("meeting_title", "Meeting")
                if p_text:
                    context_parts.append(f"[{p_title} | Segment {p_idx}]\n{p_text}")
                    sources.append({
                        "text": p_text,
                        "chunk_index": p_idx,
                        "meeting_title": p_title,
                    })
    except Exception as exc:
        logger.warning(f"[QueryEngine] Vector search warning: {exc}. Proceeding with transcript context.")

    # 2. If vector search yielded few/no sources and in_memory_transcript provided
    if not context_parts and in_memory_transcript:
        logger.info("[QueryEngine] Using in-memory transcript for direct grounding.")
        # If short transcript (< 4000 chars), use whole transcript
        if len(in_memory_transcript) <= 4000:
            context_parts.append(in_memory_transcript)
            sources = _extract_heuristic_sources(in_memory_transcript, question, top_n=3)
        else:
            heuristic_src = _extract_heuristic_sources(in_memory_transcript, question, top_n=5)
            for src in heuristic_src:
                context_parts.append(src["text"])
                sources.append(src)

    if not context_parts:
        if in_memory_transcript:
            context_parts.append(in_memory_transcript[:4000])

    if not context_parts:
        return {
            "answer": "I could not find any relevant meeting discussions or transcript records matching your question.",
            "sources": [],
            "job_id": job_id or "unknown"
        }

    context_text = "\n\n---\n\n".join(context_parts)
    title_display = f"'{meeting_title}'" if meeting_title else "the meeting recording"

    prompt = f"""You are SecureGPT's Intelligent Meeting Analyst & Assistant.
Your task is to answer the user's question accurately, concisely, and factually based on {title_display}.

RULES:
1. Base your answer strictly on the meeting context provided below.
2. If specific details (such as names, action items, dates, deadlines, or decisions) are mentioned, highlight them clearly.
3. If the answer cannot be determined from the context, state clearly what is known and that the specific detail was not discussed.
4. Format your answer with clear bullet points or bold text where appropriate for readability.

--- MEETING CONTEXT & TRANSCRIPT EXCERPTS ---
{context_text}

--- USER QUESTION ---
{question}

--- INTELLIGENT ANSWER ---
"""

    try:
        answer = generate(prompt).strip()
        # Clean up any leftover prompt tags
        if "--- INTELLIGENT ANSWER ---" in answer:
            answer = answer.split("--- INTELLIGENT ANSWER ---")[-1].strip()

        return {
            "answer": answer,
            "sources": sources,
            "job_id": job_id or "latest"
        }
    except Exception as exc:
        logger.error(f"[QueryEngine] LLM generation failed: {exc}")
        return {
            "answer": f"Error generating answer: {exc}",
            "sources": sources,
            "job_id": job_id or "unknown"
        }

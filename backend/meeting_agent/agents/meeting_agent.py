"""
Meeting Agent — Core LangGraph Nodes
--------------------------------------
Four sequential nodes that analyse the diarized transcript:

  1. rag_context_node     — pull relevant enterprise context from Qdrant
  2. summarize_node       — concise meeting summary
  3. decisions_node       — extract formal decisions as structured JSON
  4. action_items_node    — extract action items with owner + deadline
"""
import json
import logging

from retrieval_pipeline.llm.provider import generate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(response: str, fallback):
    """Strip markdown code fences then JSON-parse. Return fallback on error."""
    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Node 1 — RAG Context (optional Qdrant lookup)
# ---------------------------------------------------------------------------

def rag_context_node(state: dict) -> dict:
    """
    Extract key technical terms from the transcript and look them up
    in the enterprise Qdrant store to enrich the minutes with verified data.
    """
    transcript = state.get("diarized_transcript") or state.get("raw_transcript", "")
    if not transcript:
        state["rag_context"] = ""
        return state

    extract_prompt = f"""You are an enterprise terminology extractor.
From the following meeting transcript, extract up to 5 specific technical terms,
asset IDs, equipment names, or system names that would benefit from being
looked up in a corporate knowledge base.

Respond ONLY with a JSON object: {{"terms": ["term1", "term2", ...]}}

Transcript excerpt (first 2000 chars):
{transcript[:2000]}

JSON Output:"""

    extract_resp = generate(extract_prompt)
    data = _safe_json(extract_resp, {"terms": []})
    terms = data.get("terms", [])

    if not terms:
        state["rag_context"] = ""
        return state

    try:
        from retrieval_pipeline.retrieval import retrieve
        combined_context_parts = []
        seen = set()
        for term in terms[:5]:
            chunks = retrieve(
                query=term,
                role=state.get("user_role", "viewer"),
                department=state.get("user_department"),
                top_k=3,
            )
            for chunk in chunks:
                h = hash(chunk.text)
                if h not in seen:
                    combined_context_parts.append(f"[{term}]\n{chunk.text}")
                    seen.add(h)
        state["rag_context"] = "\n\n---\n\n".join(combined_context_parts)
    except Exception as exc:
        logger.warning(f"[MeetingAgent/RAG] Qdrant lookup failed: {exc}. Continuing without context.")
        state["rag_context"] = ""

    return state


# ---------------------------------------------------------------------------
# Node 2 — Summarize
# ---------------------------------------------------------------------------

def summarize_node(state: dict) -> dict:
    """
    Produce a concise, structured summary of the meeting.
    """
    transcript = state.get("diarized_transcript") or state.get("raw_transcript", "")
    rag_ctx = state.get("rag_context", "")
    title = state.get("meeting_title", "Internal Meeting")

    prompt = f"""You are a professional meeting analyst for a secure enterprise system.

Your task: Write a concise meeting summary (3–6 sentences) for the following transcript.
Capture: the main topic, key discussion points, and overall outcome.
Do NOT invent information not present in the transcript.
Do NOT include speaker labels in the summary — write in third person.

Meeting Title: {title}

{"Enterprise Context (verified reference data):" + chr(10) + rag_ctx + chr(10) if rag_ctx else ""}
Transcript:
{transcript}

Summary:"""

    summary = generate(prompt).strip()
    state["meeting_summary"] = summary
    logger.info("[MeetingAgent] Summary generated.")
    return state


# ---------------------------------------------------------------------------
# Node 3 — Decisions
# ---------------------------------------------------------------------------

def decisions_node(state: dict) -> dict:
    """
    Extract all formal decisions made during the meeting.
    """
    transcript = state.get("diarized_transcript") or state.get("raw_transcript", "")

    prompt = f"""You are a formal minutes analyst.

Extract ALL decisions made during the meeting transcript below.
A decision is any agreed-upon course of action, resolution, or commitment.

Respond ONLY with a JSON object:
{{
  "decisions": [
    {{
      "decision": "Clear statement of the decision",
      "rationale": "Why this decision was made (empty string if not stated)"
    }}
  ]
}}

If no decisions were made, return: {{"decisions": []}}

Transcript:
{transcript}

JSON Output:"""

    response = generate(prompt)
    data = _safe_json(response, {"decisions": []})
    decisions = data.get("decisions", [])

    normalised = []
    for d in decisions:
        if isinstance(d, dict) and d.get("decision"):
            normalised.append({
                "decision": str(d.get("decision", "")),
                "rationale": str(d.get("rationale", "")),
            })

    state["decisions"] = normalised
    logger.info(f"[MeetingAgent] {len(normalised)} decision(s) extracted.")
    return state


# ---------------------------------------------------------------------------
# Node 4 — Action Items
# ---------------------------------------------------------------------------

def action_items_node(state: dict) -> dict:
    """
    Extract actionable tasks with owner, deadline, and priority.
    """
    transcript = state.get("diarized_transcript") or state.get("raw_transcript", "")

    prompt = f"""You are a task extraction specialist for enterprise meeting minutes.

From the meeting transcript below, extract ALL action items — tasks that someone
needs to do as a result of the meeting.

For each action item, identify:
- task: clear description of what needs to be done
- owner: name or role of the responsible person ("Unknown" if not stated)
- deadline: specific date, relative time ("24 hours", "next week"), or "Not specified"
- priority: "High" if urgent/safety-critical, "Medium" if normal, "Low" otherwise

Respond ONLY with a JSON object:
{{
  "action_items": [
    {{
      "task": "...",
      "owner": "...",
      "deadline": "...",
      "priority": "High|Medium|Low"
    }}
  ]
}}

If no action items exist, return: {{"action_items": []}}

Transcript:
{transcript}

JSON Output:"""

    response = generate(prompt)
    data = _safe_json(response, {"action_items": []})
    items = data.get("action_items", [])

    normalised = []
    for item in items:
        if isinstance(item, dict) and item.get("task"):
            normalised.append({
                "task": str(item.get("task", "")),
                "owner": str(item.get("owner", "Unknown")),
                "deadline": str(item.get("deadline", "Not specified")),
                "priority": str(item.get("priority", "Medium")),
            })

    state["action_items"] = normalised
    logger.info(f"[MeetingAgent] {len(normalised)} action item(s) extracted.")
    return state

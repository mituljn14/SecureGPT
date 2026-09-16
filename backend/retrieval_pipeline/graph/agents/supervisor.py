import json
import re
from qdrant_client.http import models as qmodels

from retrieval_pipeline.llm.provider import generate, get_network_target
from retrieval_pipeline.graph.telemetry import log_event
from retrieval_pipeline.retrieval.embedding import embed_query_hybrid
from retrieval_pipeline.retrieval.filter_builder import build_filter
from retrieval_pipeline.retrieval.qdrant_client import get_client
from retrieval_pipeline.retrieval.config import DENSE_RELEVANCE_THRESHOLD
from retrieval_pipeline.retrieval.identifier_lookup import has_identifier_match
from retrieval_pipeline.retrieval.config import COMPANY_NAME, COMPANY_ALIASES

from retrieval_pipeline.retrieval.config import QDRANT_COLLECTION as COLLECTION_NAME


def _corpus_has_real_match(query: str, role: str, department) -> bool:
    """Safety net for the single supervisor LLM call above: before letting
    it answer a query directly (its own judgment that this is general
    knowledge / casual chat, NOT about the company), check whether the
    corpus actually has a strong match for it.

    The classification prompt lists examples ("Nova / NAES / our
    company"), but a private company's real query vocabulary is
    unenumerable -- a product codename, a department acronym, someone's
    name, an informally-phrased question -- any of which a single
    classification call can misjudge as "general knowledge" and then
    answer ungrounded, with no RAG context at all. That is a worse
    failure than the RRF-confidence bug fixed earlier: there, a wrong
    answer at least cited some (irrelevant) context; here the LLM would
    fabricate from nothing and sound just as confident. Same dense-cosine
    relevance probe used everywhere else in this pipeline (not the RRF
    fused score -- rank-based, looks confident even with no real match).
    """
    # A record code the caller is cleared to read is a corpus match by
    # definition -- no embedding needed, and the dense probe below would
    # score it BELOW the floor anyway (a code is not a sentence).
    if has_identifier_match(query, role, department):
        return True

    try:
        dense, _ = embed_query_hybrid(query)
    except Exception as e:
        print(f"[Supervisor safety-net] embedding failed, skipping override check: {e}")
        return False

    try:
        q_filter = build_filter(role=role, department=department, metadata_filters={})
        client = get_client()
        probe = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense, using="dense", query_filter=q_filter, limit=1,
        )
    except Exception as e:
        print(f"[Supervisor safety-net] corpus probe failed, skipping override check: {e}")
        return False

    return bool(probe.points) and probe.points[0].score >= DENSE_RELEVANCE_THRESHOLD

SUPERVISOR_PROMPT = f"""You are a routing supervisor for SecureGPT, an enterprise AI assistant for {COMPANY_NAME} (also known as {" / ".join(COMPANY_ALIASES)}).

You have access to an internal knowledge base containing the company's documents (annual reports, meeting minutes, financial data, operational records, etc.).

Your ONLY job is to classify the user's query:

1. If the query requires MATHEMATICS, CALCULATIONS (sum, average, total, revenue math), DATA ANALYSIS, CODE GENERATION, or CREATING FILES (csv, excel, pdf, reports) → route to "coding".
2. If the query is about READING AN IMAGE, PHOTO, SCAN, DIAGRAM, CAD DRAWING, BLUEPRINT, CHART IMAGE, or a SCANNED/visual PDF → route to "vision".
3. If the query is about a MEETING RECORDING, AUDIO, TRANSCRIPT, or MINUTES → route to "meeting".
4. If the query is about {COMPANY_ALIASES[0]} / "our company" / "the company" and asks for facts, summaries, or insights without complex math → route to "enterprise".
5. If the query is about OTHER companies (Tesla, Apple, Google, etc.) or general knowledge → answer directly.
6. If the query is casual chat → answer directly.

DEPARTMENTS:
- "coding": MUST BE USED for ANY math, arithmetic, data analysis, spreadsheet processing, calculations, code generation, or file creation.
- "vision": Reading or interpreting visual content -- images, scans, diagrams, CAD blueprints, chart pictures, OCR.
- "meeting": Meeting audio -- transcription, minutes, decisions, action items.
- "enterprise": General RAG retrieval for facts, insights, strategies, policies, and information extraction.

RESPONSE FORMAT:
- If routing to a department: respond with ONLY {{"department": "<name>"}}
- If answering directly: just respond naturally as a helpful assistant. No JSON.
"""

# Where the graph can send a query. Legacy names from the three-agent era
# still come back from the model occasionally; they all meant "enterprise".
_DESTINATIONS = {"enterprise", "coding", "vision", "meeting"}
_LEGACY_DESTINATIONS = {"operations": "enterprise", "technology": "enterprise", "executive": "enterprise"}
_DESTINATION_RE = re.compile(
    '"(' + "|".join(sorted(_DESTINATIONS | set(_LEGACY_DESTINATIONS))) + ')"', re.IGNORECASE
)


# Sentinel from _parse_routing: the reply was JSON (so not an answer) but
# named no department the graph knows.
UNRESOLVED = "__unresolved__"


def _as_destination(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in _DESTINATIONS:
        return v
    return _LEGACY_DESTINATIONS.get(v)


def _parse_routing(response: str):
    """Destination named by a JSON-shaped supervisor reply; None when the
    reply is prose (a direct answer); UNRESOLVED when it is JSON that
    names no known department.

    The prompt asks for {"department": "<name>"} but a 3B model does not
    always oblige: the RAGAS run (row 10) got {"enterprise": "I don't
    have the specific information..."} -- the destination as the KEY --
    which the strict parser missed, so the raw JSON was shown to the
    user as the answer. Anything JSON-shaped is a routing attempt, never
    an answer, so this resolves the destination from the canonical key,
    then any value, then any key, then a quoted name in malformed JSON,
    and finally defaults to grounded retrieval rather than echoing JSON.
    """
    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    if not text.startswith("{"):
        return None

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    if isinstance(data, dict):
        for key in ("department", "route", "destination", "agent"):
            dest = _as_destination(data.get(key))
            if dest:
                return dest
        for value in data.values():
            dest = _as_destination(value)
            if dest:
                return dest
        for key in data:
            dest = _as_destination(key)
            if dest:
                return dest

    m = _DESTINATION_RE.search(text)
    if m:
        return _as_destination(m.group(1))

    # JSON-shaped but naming no department ({"department": "other"} /
    # "general" / "none"): the model is saying "not one of mine". That is
    # a direct-answer decision, just without the answer text -- the
    # caller must never show this JSON to the user.
    return UNRESOLVED


# First-person references to the organisation. "our top concerns", "did we
# hit target", "the company's policy" are company questions regardless of
# how a small classifier labels them; the prompt says as much in words,
# this enforces it. Company names come from config (COMPANY_NAME /
# COMPANY_ALIASES), so nothing about the deployment is fixed here.
_COMPANY_REF = re.compile(
    "(?<![a-z])(our|ours|we|us|the company|company's|the firm|the organisation|the organization|"
    + "|".join(re.escape(a.lower()) for a in [COMPANY_NAME] + COMPANY_ALIASES)
    + ")(?![a-z])",
    re.IGNORECASE,
)


def _refers_to_company(query: str) -> bool:
    return bool(_COMPANY_REF.search(query or ""))


# Explicitly visual requests. No word-boundary escapes: the pattern is
# written with plain alternatives so it survives any escaping, and a
# substring match is what we want anyway ("bar chart", "charts").
_CHART_INTENT = re.compile(
    "chart|graph|plot|visualise|visualize|visualisation|"
    "visualization|histogram|diagram",
    re.IGNORECASE,
)

# ── Fast-path: instant greeting responses (0 LLM calls) ──
_GREETING_RESPONSES = {
    "hey": "Hey there! 👋 How can I help you today?",
    "hi": "Hi! 👋 What can I do for you?",
    "hello": "Hello! How can I assist you today?",
    "yo": "Yo! What's up? How can I help?",
    "sup": "Hey! What can I do for you today?",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Happy to help. 😊",
    "ok": "Got it! Let me know if there's anything else.",
    "okay": "Alright! What's next?",
    "bye": "Goodbye! Have a great day! 👋",
    "good morning": "Good morning! ☀️ How can I help you today?",
    "good evening": "Good evening! How can I assist you?",
    "good night": "Good night! Sleep well! 🌙",
    "how are you": "I'm doing great, thanks for asking! How can I help you today?",
    "what's up": "Not much! Just here to help. What do you need?",
    "whats up": "Not much! Just here to help. What do you need?",
    "who are you": "I'm SecureGPT, your enterprise AI assistant. I can help with operations, technology, executive insights, data analysis, and general questions!",
    "what can you do": "I can help you with enterprise operations, technology architecture, executive insights, data analysis & code generation, and general knowledge questions. Just ask away!",
}







def supervisor_node(state):
    query = state["query"]
    stripped = query.strip().rstrip("?!. ").lower()
    
    # ── TIER 1: Instant greeting response (0ms, no LLM) ──
    if stripped in _GREETING_RESPONSES:
        state["agent_destination"] = "__direct__"
        state["answer"] = _GREETING_RESPONSES[stripped]
        print(f"⚡ Supervisor INSTANT → greeting")
        log_event(state, action="Supervisor Routing", 
                  details="Instant greeting response (no LLM)", 
                  network_target="local (instant)")
        return state
    
    # ── TIER 1b: Attachment type decides, before any LLM call ──
    # What was actually attached is a fact, not a judgement call: audio can
    # only be handled by the meeting agent and an image only by the vision
    # agent, so asking a 3B model to infer it just adds a failure mode.
    uploaded = state.get("uploaded_files") or []
    if uploaded:
        from retrieval_pipeline.graph.agents.meeting import pick_audio_file
        from retrieval_pipeline.graph.agents.vision import pick_vision_file

        if pick_audio_file(uploaded):
            state["agent_destination"] = "meeting"
            print("[FAST] Supervisor -> MEETING (audio attached)")
            log_event(state, action="Supervisor Routing",
                      details="Routed to MEETING AGENT (audio file attached)",
                      network_target="local (file type)")
            return state

        if pick_vision_file(uploaded):
            state["agent_destination"] = "vision"
            print("[FAST] Supervisor -> VISION (image/PDF attached)")
            log_event(state, action="Supervisor Routing",
                      details="Routed to VISION AGENT (image/PDF attached)",
                      network_target="local (file type)")
            return state

        # A spreadsheet plus an explicitly visual request goes to vision:
        # its ChartGenerator draws from validated templates, whereas the
        # coding agent would generate and execute plotting code to reach the
        # same picture. Spreadsheets WITHOUT a chart request still go to the
        # coding agent, which is what actually computes over data.
        if _CHART_INTENT.search(query) and pick_vision_file(uploaded, include_tabular=True):
            state["agent_destination"] = "vision"
            print("[FAST] Supervisor -> VISION (chart requested for tabular file)")
            log_event(state, action="Supervisor Routing",
                      details="Routed to VISION AGENT (chart/plot requested for a spreadsheet)",
                      network_target="local (file type)")
            return state

        # A file was attached that no agent can read. Say so directly rather
        # than letting it fall through to the LLM: the enterprise agent never
        # looks at attachments, so it answers "I don't have any information
        # about the attached file" -- which reads as though the file was
        # rejected for its contents rather than its extension.
        import os as _os

        from retrieval_pipeline.graph.agents.meeting import AUDIO_EXTS
        from retrieval_pipeline.graph.agents.vision import INGESTIBLE_EXTS

        readable = AUDIO_EXTS | INGESTIBLE_EXTS
        unreadable = [
            _os.path.basename(str(f)) for f in uploaded
            if _os.path.splitext(str(f))[1].lower() not in readable
        ]
        if unreadable and len(unreadable) == len(uploaded):
            exts = ", ".join(sorted(e.lstrip(".") for e in readable))
            state["agent_destination"] = "__direct__"
            state["answer"] = (
                f"I can't read {', '.join(unreadable)} - that file type is not "
                f"handled by any agent.\n\nSupported types: {exts}."
            )
            print(f"[FAST] Supervisor -> DIRECT (unsupported attachment: {unreadable})")
            log_event(state, action="Supervisor Routing",
                      details=f"Unsupported attachment type: {', '.join(unreadable)}",
                      network_target="local (file type)")
            return state

    # ── TIER 2: Single LLM call — classify AND answer general queries ──
    prompt = f"{SUPERVISOR_PROMPT}\n\nUser: {query}\n\nYour response:"
    response = generate(prompt, model_override=state.get("model_override"))
    
    response_stripped = response.strip()

    department = _parse_routing(response_stripped)

    if department == UNRESOLVED:
        # No usable answer text to fall back on; the direct path below
        # will generate one if the corpus has nothing.
        department = None
        response_stripped = ""

    if department is None and _refers_to_company(query):
        # The model answered in prose, but the query is about *us*. A
        # direct answer here is by construction ungrounded -- send it
        # through retrieval, where "not found" is at least an honest
        # outcome. (Eval row 1: "what did the risk assessment report
        # flag as our top concerns?" was answered from nothing, five
        # runs out of five.)
        department = "enterprise"
        print("[Supervisor] prose reply to a company-referencing query -> ENTERPRISE")
    if department:
        # Enterprise query → route to department agent for RAG retrieval
        state["agent_destination"] = department
        print(f"[Supervisor LLM] -> {department.upper()}")
        log_event(state, action="Supervisor Routing", 
                  details=f"Routed query to {department.upper()} AGENT",
                  network_target=get_network_target())
    else:
        # LLM judged this general knowledge / casual chat and answered
        # directly -- but don't trust that classification blindly. If the
        # corpus actually has a strong match, this is a company question
        # the classifier missed; override and send it through grounded
        # RAG retrieval instead of an ungrounded direct answer.
        if _corpus_has_real_match(query, state["role"], state.get("department")):
            state["agent_destination"] = "enterprise"
            print(f"🧠 Supervisor LLM → DIRECT, but corpus match found → ENTERPRISE (override)")
            log_event(state, action="Supervisor Routing",
                      details="Overrode direct-answer classification: corpus contains a relevant "
                              "match for this query, routing to ENTERPRISE for grounded retrieval "
                              "instead of an ungrounded direct answer",
                      network_target=get_network_target())
        else:
            if not response_stripped:
                # The classifier declined to route but gave no prose
                # (JSON with an unknown department). One more plain call
                # to actually answer -- rare path, and the alternative
                # is showing the user a routing JSON blob.
                response_stripped = generate(
                    "Answer the user's question directly and concisely.\n\n"
                    f"User: {query}\n\nAnswer:",
                    model_override=state.get("model_override"),
                ).strip()
            state["agent_destination"] = "__direct__"
            state["answer"] = response_stripped
            print(f"🧠 Supervisor LLM → DIRECT ANSWER (1 call)")
            log_event(state, action="Supervisor Routing",
                      details="General query answered directly by supervisor (single LLM call)",
                      network_target=get_network_target())
    
    return state

"""
LangGraph Nodes

Each node performs exactly one task.
"""
import json
from retrieval_pipeline.retrieval import retrieve, retrieve_with_confidence
from retrieval_pipeline.retrieval.sparse_rerank import sparse_rerank
from retrieval_pipeline.context.builder import build_context
from retrieval_pipeline.context.prompt import SYSTEM_PROMPT, ANSWER_PROMPT
from retrieval_pipeline.llm.provider import generate, get_network_target
from retrieval_pipeline.graph.telemetry import log_event, summarize_sources, sources_text
from retrieval_pipeline.retrieval.config import QDRANT_URL, FAST_PATH_CONFIDENCE_THRESHOLD, MAX_CONTEXT_CHUNKS

# ---------------------------------------------------------
# Complexity Analyzer Node
# ---------------------------------------------------------

from retrieval_pipeline.graph.agents.coding_agent.retriever import Retriever

_coding_retriever = Retriever()

def fast_retrieval_node(state):
    query = state["query"]

    # Use true Qdrant Hybrid Search (Dense + Sparse via FastEmbed). Also get
    # a real relevance confidence back -- the raw dense cosine of the single
    # best match, NOT the fused RRF score (rank-based, ~1/(rank+k): looks
    # confident for whatever ranks #1 even when nothing relevant exists).
    # Previously this just set confidence=1.0 whenever retrieve() returned
    # *any* hits at all -- since it always returns top_k results regardless
    # of relevance, that meant confidence was effectively always 1.0 and
    # the CRAG/multi-hop escalation below almost never fired, even for
    # queries with no genuine corpus match. See DENSE_RELEVANCE_THRESHOLD
    # in retrieval/config.py for the calibration.
    try:
        hits, confidence = retrieve_with_confidence(
            query=query,
            role=state["role"],
            department=state.get("department"),
            top_k=5
        )
    except Exception as e:
        print(f"Hybrid search failed: {e}")
        hits, confidence = [], 0.0

    accepted = bool(hits) and confidence >= FAST_PATH_CONFIDENCE_THRESHOLD
    state["retrieved_chunks"] = hits
    state["retrieval_confidence"] = confidence if hits else 0.0
    state["retrieval_stage"] = "fast" if hits else ""
    state["retrieval_path"] = [{
        "stage": "fast", "score": round(confidence, 3), "chunks": len(hits), "accepted": accepted,
    }]

    exact = [c for c in hits if c.score >= 1.0]
    what = (f"Hybrid search returned {len(hits)} chunk(s); confidence {confidence:.3f} "
            f"(dense cosine of best match" + (f", {len(exact)} exact-ID hit(s)" if exact else "") + ").")
    decision = (f"≥ {FAST_PATH_CONFIDENCE_THRESHOLD:.2f} → accepted, no grading needed."
                if accepted else
                f"< {FAST_PATH_CONFIDENCE_THRESHOLD:.2f} → not trusted, escalating to CRAG.")
    srcs = summarize_sources(hits)
    log_event(state, action="Fast Hybrid Retrieval",
              details=f"{what} Documents: {sources_text(srcs)}. {decision}",
              network_target=QDRANT_URL, sources={"retrieved": srcs})
    return state

# ---------------------------------------------------------
# CRAG Retrieval Node
# ---------------------------------------------------------

# CRAG walks down the ranked result list one window at a time, grading each
# window and keeping the best. Capped at CRAG_MAX_ITERATIONS.
#
# Why a sliding window rather than a growing top_k: retrieve() returns a
# deterministic ranking, so top_k=20 has the *same* first 5 results as
# top_k=5. Re-grading a wider fetch would score identical evidence and make
# every iteration after the first a no-op. Sliding to the next window is
# what actually puts new evidence in front of the grader. CRAG_WINDOW
# matches MAX_CONTEXT_CHUNKS -- selecting more chunks than build_context()
# will forward to the LLM would be pointless.
CRAG_MAX_ITERATIONS = 3
CRAG_WINDOW = 5
CRAG_SUFFICIENT_SCORE = 0.7


def _grade_relevance(query: str, chunks, model_override) -> float:
    """LLM self-graded relevance of `chunks` to `query`, 0.0..1.0.

    Returns 0.0 when the reply cannot be parsed -- an unparseable grade must
    never read as a good one.
    """
    context_text = "\n\n".join([c.text for c in chunks])
    prompt = f"""You are an evaluator estimating the relevance of retrieved documents to the query.
Rate the relevance from 0.0 to 1.0, where 1.0 means the documents fully answer the query.
Respond ONLY with a JSON object with a single key "score" and the float value. Do not output anything else.

Query: {query}
Documents:
{context_text}

JSON Output:
"""
    response = generate(prompt, model_override=model_override)
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        data = json.loads(response)
        return float(data.get("score", 0.0))
    except Exception:
        return 0.0


def crag_retrieval_node(state):
    query = state["query"]
    model_override = state.get("model_override")

    # One retrieval deep enough to feed every iteration, rather than
    # re-querying Qdrant each pass for results already in hand.
    pool = retrieve(
        query=query,
        role=state["role"],
        department=state.get("department"),
        top_k=CRAG_WINDOW * CRAG_MAX_ITERATIONS,
    )

    best_chunks = []
    best_score = 0.0
    iterations = 0

    for i in range(CRAG_MAX_ITERATIONS):
        window = pool[i * CRAG_WINDOW:(i + 1) * CRAG_WINDOW]
        if not window:
            break

        iterations += 1
        score = _grade_relevance(query, window, model_override)

        if score > best_score:
            best_score = score
            best_chunks = window

        if score >= CRAG_SUFFICIENT_SCORE:
            break

    # Never hand an empty set forward: if this came back dry, the chunks
    # fast_retrieval_node already found are strictly better than nothing.
    # (The previous version overwrote unconditionally.)
    if best_chunks:
        state["retrieved_chunks"] = best_chunks
        state["retrieval_stage"] = "crag"

    # Graded relevance is its own signal -- deliberately NOT written over
    # retrieval_confidence, which holds the dense cosine from the fast path.
    state["crag_score"] = best_score

    accepted = best_score >= CRAG_SUFFICIENT_SCORE
    state.setdefault("retrieval_path", []).append({
        "stage": "crag", "score": round(best_score, 3), "chunks": len(best_chunks),
        "accepted": accepted, "kept_previous": not best_chunks,
    })
    what = (f"Graded {iterations} window(s) of {CRAG_WINDOW} over a pool of {len(pool)}; "
            f"best window scored {best_score:.2f}.")
    if not best_chunks:
        what += " No window scored above zero — keeping the fast-path chunks."
    decision = (f"≥ {CRAG_SUFFICIENT_SCORE:.1f} → accepted."
                if accepted else
                f"< {CRAG_SUFFICIENT_SCORE:.1f} → insufficient, escalating to multi-agent retrieval.")
    pool_srcs, win_srcs = summarize_sources(pool), summarize_sources(best_chunks)
    what += f" Pool documents: {sources_text(pool_srcs)}. Best window: {sources_text(win_srcs)}."
    log_event(state, action="CRAG Retrieval", details=f"{what} {decision}",
              network_target=get_network_target(),
              sources={"pool": pool_srcs, "selected": win_srcs})
    log_event(
        state,
        action="Database Query",
        details=f"Searched vector database for relevant contexts",
        network_target=QDRANT_URL
    )
    return state

# ---------------------------------------------------------
# Multi-Agent Retrieval Node
# ---------------------------------------------------------

def multi_agent_retrieval_node(state):
    original_query = state["query"]
    
    # 1. Query Rewrite
    prompt = f"""You are a query rewriting assistant. Generate 3 variations of the following user query to improve document retrieval. Use semantic variations (synonyms), abstraction (broader/narrower), and decomposition.
Respond ONLY with a JSON object with a key "queries" containing a list of 3 string variations. Do not output anything else.

Query: {original_query}
JSON Output:
"""
    response = generate(prompt, model_override=state.get("model_override"))
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        data = json.loads(response)
        queries = data.get("queries", [original_query])
    except Exception:
        queries = [original_query]
        
    state["rewritten_queries"] = queries
    
    # 2. Pseudo-Hybrid Search
    all_chunks = []
    seen_ids = set()
    for q in queries + [original_query]:
        chunks = retrieve(
            query=q,
            role=state["role"],
            department=state.get("department"),
            top_k=20  # fetch larger pool
        )
        for chunk in chunks:
            chunk_hash = hash(chunk.text)
            if chunk_hash not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk_hash)
                
    if not all_chunks:
        # Leave whatever fast_retrieval/CRAG already found in place rather
        # than blanking it -- this node is last in the cascade, so wiping
        # state here would send the enterprise agent an empty context.
        state.setdefault("retrieval_path", []).append({
            "stage": "multi_agent", "score": None, "chunks": 0,
            "accepted": False, "kept_previous": True,
        })
        log_event(state, action="Multi-Agent Retrieval",
                  details=(f"Rewrote the query into {len(queries)} variant(s) but retrieval "
                           f"returned nothing readable by this role. Keeping the previous stage's chunks."),
                  network_target=get_network_target())
        return state
        
    # Lexical reranking against the BGE-M3 sparse weights already stored in
    # Qdrant -- the same representation the main hybrid path ranks on.
    # (Was BM25Okapi over the candidate pool, whose IDF was derived from
    # those few dozen chunks and so carried no real signal; see
    # retrieval/sparse_rerank.py.)
    top_chunks = sparse_rerank(original_query, all_chunks, top_k=10)
    
    # 3. Evidence Verification (Filter contradictory/irrelevant)
    verified_chunks = []
    for chunk in top_chunks:
        verify_prompt = f"""Does the following document contain information relevant and consistent to answering the query?
Respond ONLY with a JSON object with a key "valid" and a boolean true/false.

Query: {original_query}
Document: {chunk.text}

JSON Output:
"""
        response = generate(verify_prompt, model_override=state.get("model_override"))
        is_valid = True
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            data = json.loads(response)
            is_valid = bool(data.get("valid", True))
        except Exception:
            pass
        if is_valid:
            verified_chunks.append(chunk)
            
    # 4. Multi-Hop Evaluation Loop
    hops = 0
    while hops < 2:  # caps at max 3 hops total (1 initial + 2 loop)
        eval_prompt = f"""Evaluate the current evidence to answer the query. Is the evidence sufficient?
If not sufficient, suggest ONE new query to find missing information.
Respond ONLY with a JSON object: {{"sufficient": bool, "next_query": "string or null"}}

Query: {original_query}
Evidence:
{"\n\n".join([c.text for c in verified_chunks[:5]])}

JSON Output:
"""
        response = generate(eval_prompt, model_override=state.get("model_override"))
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            data = json.loads(response)
            sufficient = data.get("sufficient", True)
            next_query = data.get("next_query")
        except Exception:
            sufficient = True
            next_query = None
            
        if sufficient or not next_query:
            break
            
        hops += 1
        new_chunks = retrieve(
            query=next_query,
            role=state["role"],
            department=state.get("department"),
            top_k=5
        )
        for chunk in new_chunks:
            chunk_hash = hash(chunk.text)
            if chunk_hash not in seen_ids:
                verified_chunks.append(chunk)
                seen_ids.add(chunk_hash)

    # 5. Final Reranking (Keep top 5 verified chunks)
    # Only overwrite if verification actually left something. If every
    # candidate was rejected, the CRAG window already in state is better
    # evidence than none at all.
    if verified_chunks:
        state["retrieved_chunks"] = verified_chunks[:5]
        state["retrieval_stage"] = "multi_agent"

    state.setdefault("retrieval_path", []).append({
        "stage": "multi_agent", "score": None, "chunks": len(verified_chunks[:5]),
        "accepted": bool(verified_chunks), "kept_previous": not verified_chunks,
    })
    what = (f"Rewrote the query into {len(queries)} variant(s), retrieved {len(all_chunks)} candidates, "
            f"lexically reranked to {len(top_chunks)}, LLM-verified {len(verified_chunks)}.")
    decision = ("Using the verified chunks." if verified_chunks
                else "Every candidate was rejected — keeping the previous stage's chunks.")
    cand_srcs, rr_srcs, ver_srcs = (summarize_sources(all_chunks), summarize_sources(top_chunks),
                                    summarize_sources(verified_chunks[:5]))
    what += (f" Candidates: {sources_text(cand_srcs)}. Reranked: {sources_text(rr_srcs)}. "
             f"Verified: {sources_text(ver_srcs)}.")
    log_event(state, action="Multi-Agent Retrieval", details=f"{what} {decision}",
              network_target=get_network_target(),
              sources={"candidates": cand_srcs, "reranked": rr_srcs, "verified": ver_srcs})
    log_event(
        state, 
        action="Database Query", 
        details=f"Searched vector database for relevant contexts", 
        network_target=QDRANT_URL
    )
    return state

# ---------------------------------------------------------
# Context Builder Node
# ---------------------------------------------------------

STAGE_LABELS = {"fast": "fast path", "crag": "CRAG", "multi_agent": "multi-agent"}


def context_builder_node(state):
    context = build_context(
        state.get("retrieved_chunks", [])
    )
    state["context"] = context

    stage = state.get("retrieval_stage") or ""
    path = state.get("retrieval_path") or []
    n = len(state.get("retrieved_chunks", []))
    steps = []
    for p in path:
        score = "" if p.get("score") is None else f" {p['score']:.2f}"
        mark = "✓" if p.get("accepted") else "✗"
        steps.append(f"{STAGE_LABELS.get(p['stage'], p['stage'])}{score} {mark}")
    source = STAGE_LABELS.get(stage, stage or "no retrieval")
    details = f"Context: {n} chunk(s) from {source}."
    if steps:
        details += " Path: " + " → ".join(steps) + "."
    used = summarize_sources(state.get("retrieved_chunks", []), with_ids=True, limit=MAX_CONTEXT_CHUNKS)
    details += f" Documents: {sources_text(used)}."
    log_event(state, action="Built Context", details=details, network_target="Local CPU",
              sources={"context": used})
    # Machine-readable copy on the same trace event, so the UI never parses text.
    state["trajectory_log"][-1]["retrieval_stage"] = stage
    state["trajectory_log"][-1]["retrieval_path"] = path
    return state

# ---------------------------------------------------------
# LLM Generation Node
# ---------------------------------------------------------

def generation_node(state):
    prompt = f"""
{SYSTEM_PROMPT}

{ANSWER_PROMPT.format(
query=state["query"],
context=state.get("context", "")
)}
"""
    print("=" * 80)
    print("Prompt Sent To LLM")
    print("=" * 80)
    print(prompt)

    answer = generate(prompt, model_override=state.get("model_override"))

    state["prompt"] = prompt
    state["answer"] = answer
    return state
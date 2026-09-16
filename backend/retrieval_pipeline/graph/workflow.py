"""
SecureGPT LangGraph Workflow
"""
from langgraph.graph import StateGraph, END

from retrieval_pipeline.graph.state import GraphState
from retrieval_pipeline.graph.nodes import (
    fast_retrieval_node,
    crag_retrieval_node,
    multi_agent_retrieval_node,
    context_builder_node,
    CRAG_SUFFICIENT_SCORE,
)
from retrieval_pipeline.graph.agents.supervisor import supervisor_node
from retrieval_pipeline.graph.agents.enterprise import enterprise_agent_node
from retrieval_pipeline.graph.agents.general import general_agent_node
from retrieval_pipeline.graph.agents.coding import coding_agent_node
from retrieval_pipeline.graph.agents.vision import vision_agent_node
from retrieval_pipeline.graph.agents.meeting import meeting_agent_node
from retrieval_pipeline.retrieval.config import DENSE_RELEVANCE_THRESHOLD, FAST_PATH_CONFIDENCE_THRESHOLD
from retrieval_pipeline.graph.telemetry import traced_node

# ---------------------------------------------------------

builder = StateGraph(GraphState)

# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------
# Every node is wrapped by traced_node: the session log gets a node_enter /
# node_exit pair with timing, and every trace event emitted inside carries
# the node that produced it. Graph wiring below is otherwise unchanged.
for _name, _fn in [
    ("supervisor", supervisor_node),
    ("fast_retrieval", fast_retrieval_node),
    ("crag_retrieval", crag_retrieval_node),
    ("multi_agent_retrieval", multi_agent_retrieval_node),
    ("build_context", context_builder_node),
    ("enterprise", enterprise_agent_node),
    ("coding", coding_agent_node),
    ("vision", vision_agent_node),
    ("meeting", meeting_agent_node),
    ("general", general_agent_node),
]:
    builder.add_node(_name, traced_node(_name, _fn))

# ---------------------------------------------------------
# Edges
# ---------------------------------------------------------
# Supervisor is the entry point — it classifies the query
builder.set_entry_point("supervisor")

def route_supervisor(state: GraphState) -> str:
    """Route based on supervisor decision.
    
    - __direct__: Supervisor already answered (general/greeting) → END immediately
    - general: Needs the general agent with possible context
    - coding: Goes straight to coding agent sandbox
    - enterprise: Need RAG retrieval first
    """
    dest = state.get("agent_destination", "general")
    
    if dest == "__direct__":
        # Supervisor already set state["answer"] — skip everything
        return "__end__"
    elif dest == "enterprise":
        return "fast_retrieval"
    elif dest == "coding":
        return "coding"
    elif dest == "vision":
        # Vision runs its own retrieval inside the pipeline, so it does not
        # enter the enterprise cascade.
        return "vision"
    elif dest == "meeting":
        return "meeting"
    else:
        return "general"

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "__end__": END,
        "fast_retrieval": "fast_retrieval",
        "coding": "coding",
        "vision": "vision",
        "meeting": "meeting",
        "general": "general"
    }
)

# Retrieval cascade (only for enterprise department queries)
#
# retrieval_confidence (set by fast_retrieval_node) is now the raw dense
# cosine of the single best match -- a real 0..1 similarity, not the old
# hardcoded 1.0-whenever-anything-came-back value. DENSE_RELEVANCE_THRESHOLD
# (0.50, retrieval/config.py) is the "is this even real" floor shared with
# the coding agent; empirically, genuine corpus matches land noticeably
# above that, around 0.59-0.63+. FAST_PATH_CONFIDENCE_THRESHOLD sits
# between the two so a genuinely good match is trusted via the cheap fast
# path, while anything weak or borderline gets a second, more thorough
# look via CRAG/multi-hop/query-rewriting instead of a possibly-wrong
# confident-looking answer.
# FAST_PATH_CONFIDENCE_THRESHOLD lives in retrieval/config.py (env-tunable).
assert FAST_PATH_CONFIDENCE_THRESHOLD >= DENSE_RELEVANCE_THRESHOLD, (
    "FAST_PATH_CONFIDENCE_THRESHOLD must stay >= DENSE_RELEVANCE_THRESHOLD -- "
    "trusting the fast path should never be easier than clearing the basic "
    "relevance floor."
)

# Escalation runs cheapest-first:
#
#   fast_retrieval  -> 1 embed + 1 hybrid query, 0 LLM calls
#   crag_retrieval  -> 1 hybrid query + up to CRAG_MAX_ITERATIONS grades
#   multi_agent     -> query rewriting, per-chunk verification, multi-hop
#
# Each stage is entered only when the one before it could not produce
# evidence good enough to answer from, so the expensive path is reached
# only when it is actually needed.
#
# The previous wiring went fast -> multi_agent -> crag, and crag ended by
# overwriting retrieved_chunks unconditionally with a plain retrieve() of
# the original query -- discarding everything multi_agent had just spent
# ~13 LLM calls producing, and landing on the same chunks the fast path had
# already rejected. Ordering it cheapest-first and making each stage
# additive fixes both halves of that.
def route_fast_retrieval(state: GraphState) -> str:
    confidence = state.get("retrieval_confidence", 0.0)
    if confidence >= FAST_PATH_CONFIDENCE_THRESHOLD:
        return "build_context"
    return "crag_retrieval"

builder.add_conditional_edges(
    "fast_retrieval",
    route_fast_retrieval,
    {
        "build_context": "build_context",
        "crag_retrieval": "crag_retrieval"
    }
)

# CRAG grades its own evidence, so its own sufficiency score decides whether
# the agentic path is needed. This is CRAG_SUFFICIENT_SCORE from nodes.py --
# imported rather than re-declared so the two cannot drift apart.
def route_crag(state: GraphState) -> str:
    if state.get("crag_score", 0.0) >= CRAG_SUFFICIENT_SCORE:
        return "build_context"
    return "multi_agent_retrieval"

builder.add_conditional_edges(
    "crag_retrieval",
    route_crag,
    {
        "build_context": "build_context",
        "multi_agent_retrieval": "multi_agent_retrieval"
    }
)

# Last stage in the cascade -- nothing may overwrite its output.
builder.add_edge("multi_agent_retrieval", "build_context")

def route_department(state: GraphState) -> str:
    """Route from build_context to the specific RAG department agent."""
    dest = state.get("agent_destination", "general")
    if dest != "enterprise":
        dest = "general"
    return dest

builder.add_conditional_edges(
    "build_context",
    route_department,
    {
        "enterprise": "enterprise",
        "general": "general"
    }
)

builder.add_edge("enterprise", END)
builder.add_edge("coding", END)
builder.add_edge("vision", END)
builder.add_edge("meeting", END)
builder.add_edge("general", END)

# ---------------------------------------------------------
graph = builder.compile()
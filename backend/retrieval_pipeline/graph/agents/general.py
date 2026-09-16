from retrieval_pipeline.llm.provider import generate, get_network_target
from retrieval_pipeline.graph.telemetry import log_event

GENERAL_SYSTEM_PROMPT = """You are SecureGPT, an enterprise AI assistant.
You are helpful, friendly, and conversational.
If enterprise context is provided below, answer based on that context.
If no context is provided, respond naturally as a helpful assistant. Do not fabricate enterprise data."""

def general_agent_node(state):
    context = state.get("context", "")
    query = state["query"]

    if context and context.strip():
        prompt = f"""{GENERAL_SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""
    else:
        # No retrieval context — direct conversational response
        prompt = f"""{GENERAL_SYSTEM_PROMPT}

Question:
{query}

Answer:
"""

    print("=" * 80)
    print("Routed to: GENERAL AGENT")
    print(f"  Context present: {bool(context and context.strip())}")
    print("=" * 80)
    
    answer = generate(prompt, model_override=state.get("model_override"))
    state["prompt"] = prompt
    state["answer"] = answer
    log_event(
        state, 
        action="GENERAL Agent", 
        details="Generated final answer" + (" using context" if context else " (direct chat)"), 
        network_target=get_network_target()
    )
    return state

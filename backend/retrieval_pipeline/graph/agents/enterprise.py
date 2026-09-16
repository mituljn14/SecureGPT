from retrieval_pipeline.llm.provider import generate, get_network_target
from retrieval_pipeline.graph.telemetry import log_event

ENTERPRISE_SYSTEM_PROMPT = """You are an Enterprise AI Assistant for Nova Automobile Engineering Solutions Pvt. Ltd. (also known as Nova Engineering / NAES).
You have access to the company's internal knowledge base. 
When answering, prioritize conciseness, actionable insights, and accuracy.
Always answer ONLY using the provided enterprise context. If the answer is not in the context, say so.
Do not fabricate information.
"""

def enterprise_agent_node(state):
    context = state.get("context", "No enterprise context available.")
    prompt = f"""{ENTERPRISE_SYSTEM_PROMPT}

Context:
{context}

Question:
{state["query"]}

Answer as the Enterprise Assistant:
"""
    print("=" * 80)
    print("Routed to: ENTERPRISE AGENT")
    print("=" * 80)
    
    answer = generate(prompt, model_override=state.get("model_override"))
    state["prompt"] = prompt
    state["answer"] = answer
    log_event(
        state, 
        action="ENTERPRISE Agent", 
        details="Generated final answer using context", 
        network_target=get_network_target()
    )
    return state

from retrieval_pipeline.graph.telemetry import log_event
from retrieval_pipeline.llm.provider import get_network_target
from retrieval_pipeline.graph.agents.coding_agent.agent_graph import build_graph, run_agent_query
import sys
import os

# Build the coding agent subgraph once
_coding_app = build_graph()

def coding_agent_node(state):
    query = state["query"]
    
    print("=" * 80)
    print("Routed to: CODING AGENT")
    print("=" * 80)
    
    # Save the original working directory
    original_cwd = os.getcwd()
    try:
        # Move into the coding_agent directory so file_writer.py's relative
        # output paths (data/outputs/) resolve consistently regardless of
        # where the FastAPI process was launched from.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        coding_agent_dir = os.path.join(current_dir, "coding_agent")
        if os.path.exists(coding_agent_dir):
            os.chdir(coding_agent_dir)

        final_coding_state = run_agent_query(
            _coding_app,
            query,
            role=state["role"],
            department=state.get("department"),
            uploaded_files=state.get("uploaded_files", []),
            model_override=state.get("model_override")
        )
        
    finally:
        os.chdir(original_cwd)
    
    final_answer = final_coding_state.get("final_answer", "Coding agent failed to produce an answer.")
    state["generated_files"] = final_coding_state.get("generated_files", [])
    # The python the agent wrote, so the UI can show it verbatim instead of
    # asking the user to trust an unexplained number.
    state["generated_code"] = final_coding_state.get("generated_code")
    # The verified JSON the sandbox printed: {"answer": ..., "details": {...}}.
    # Surfacing `details` lets the UI show real computed figures as metric
    # cards instead of the frontend re-parsing prose for numbers.
    _v = final_coding_state.get("verification")
    # node_verify stores VerificationResult.to_dict(), so this is a dict;
    # tolerate the dataclass too in case a caller hands one through.
    _result = _v.get("result") if isinstance(_v, dict) else getattr(_v, "result", None)
    if isinstance(_result, dict):
        state["structured"] = _result

    state["prompt"] = "Routed to standalone coding agent."
    state["answer"] = final_answer

    # Merge the coding agent's OWN per-step trajectory (retrieval, code
    # generation, sandbox execution -- each logged with its real network
    # target, see agent_graph.py) into the outer trajectory, instead of
    # collapsing everything into one hardcoded "Docker Sandbox / Ollama"
    # line. That single label was wrong on two counts: it always said
    # "Ollama" even when LLM_DEPLOYMENT_TYPE was "cloud" (so a demo
    # switching to a real external API would still show "local" forever),
    # and it merged two structurally different things -- an LLM call
    # (network-reachable, backend-dependent) and a --network-none Docker
    # container (structurally offline no matter what) -- into one line,
    # so neither could be verified independently.
    inner_trajectory = final_coding_state.get("trajectory_log", [])
    state.setdefault("trajectory_log", []).extend(inner_trajectory)

    log_event(
        state,
        action="CODING Agent",
        details=f"Coding agent finished ({len(inner_trajectory)} step(s) above).",
        network_target=None,
    )
    return state

import os

DEPLOYMENT_TYPE = os.getenv("LLM_DEPLOYMENT_TYPE", "on_premise").lower()

if DEPLOYMENT_TYPE == "cloud":
    from .cloud_handler import generate as _generate
    def get_network_target():
        return f"Cloud API (Model: {os.getenv('CLOUD_MODEL', 'gpt-4o-mini')})"
elif DEPLOYMENT_TYPE == "on_premise":
    from .on_premise_handler import generate as _generate
    def get_network_target():
        return os.getenv("ON_PREMISE_API_BASE", "http://localhost:11434/v1")
else:
    raise ValueError(f"Unknown LLM deployment type: {DEPLOYMENT_TYPE}")

from retrieval_pipeline.graph.logger import log_detailed_session

def generate(prompt: str, model_override: str = None, max_tokens: int = None) -> str:
    target = get_network_target()
    log_detailed_session("llm_request", {"prompt": prompt, "network_target": target, "model_override": model_override})
    try:
        response = _generate(prompt, model_override=model_override, max_tokens=max_tokens)
        log_detailed_session("llm_response", {"response": response})
        return response
    except Exception as e:
        log_detailed_session("llm_error", {"error": str(e)})
        raise e
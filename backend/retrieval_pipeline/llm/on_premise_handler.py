import os
from litellm import completion

def generate(prompt: str, model_override: str = None, max_tokens: int = None) -> str:
    """
    Handles generation for Enterprise On-Premise models (vLLM, Ollama, OpenAI-compatible).
    Passes custom API Base URL and Keys if required.
    """
    default_model = os.getenv("ON_PREMISE_MODEL", "openai/qwen2.5-coder:3b")
    # If a specific open-source model string is selected by user, use it; otherwise fallback to default
    if model_override and model_override != "Auto (Supervisor)":
        if not model_override.startswith("openai/"):
            model_override = f"openai/{model_override}"
        model_name = model_override
    else:
        model_name = default_model
        
    api_base = os.getenv("ON_PREMISE_API_BASE", "http://localhost:11434/v1")
    api_key = os.getenv("ON_PREMISE_API_KEY", "optional")
    
    kwargs = dict(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        api_base=api_base,
        api_key=api_key,
        temperature=0.0
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    response = completion(**kwargs)
    
    return response.choices[0].message.content

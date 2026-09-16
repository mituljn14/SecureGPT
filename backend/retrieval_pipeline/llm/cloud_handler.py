import os
from litellm import completion

def generate(prompt: str, model_override: str = None, max_tokens: int = None) -> str:
    """
    Handles generation for Private Cloud APIs using LiteLLM.
    LiteLLM automatically picks up the OPENAI_API_KEY, GEMINI_API_KEY, etc. 
    from the environment based on the model prefix.
    """
    default_model = os.getenv("CLOUD_MODEL", "gpt-4o-mini")
    if model_override and model_override != "Auto (Supervisor)":
        model_name = model_override
    else:
        model_name = default_model
    
    kwargs = dict(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    response = completion(**kwargs)
    
    return response.choices[0].message.content

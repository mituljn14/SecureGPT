# pipeline/chunking/tokens.py
"""Single shared tokenizer for the whole chunking stage.

Uses tiktoken's cl100k_base as a fast proxy for BGE-M3's XLM-RoBERTa
tokenizer. They are not identical, but for *budgeting* this is safe:
our hard ceiling is 1024 tokens while BGE-M3 accepts 8192, so even a 2x
underestimate cannot overflow the model. Using the real XLM-R tokenizer
would mean loading transformers in every worker process for no practical
gain at these sizes.
"""
from functools import lru_cache

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))

    def encode(text: str):
        return _ENC.encode(text, disallowed_special=())

    def decode(ids) -> str:
        return _ENC.decode(ids)

except Exception:  # pragma: no cover - fallback if tiktoken unavailable
    _ENC = None

    def count_tokens(text: str) -> int:
        return int(len(text.split()) / 0.75) + 1

    def encode(text: str):
        return text.split()

    def decode(ids) -> str:
        return " ".join(ids)

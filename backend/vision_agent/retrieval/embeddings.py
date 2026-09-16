"""
Embeddings -- delegates to the single v2 embedder.

This module previously loaded its own BGE-M3 instance. Two embedders in
one process is both wasteful (a second multi-GB model in memory) and
dangerous: "1024-dim BGE-M3" produced by a different runtime is not
guaranteed to share a vector space with the corpus, and the failure mode
is silent -- plausible-looking results, not an exception.

`embed_query_hybrid` is the same function the enterprise retrieval path
and the coding agent use.
"""
from __future__ import annotations

from typing import List

from retrieval_pipeline.retrieval.embedding import embed_query_hybrid


class EmbeddingGenerator:
    """Dense-vector interface kept for backward compatibility."""

    def __init__(self, *_args, **_kwargs):
        pass

    def embed_query(self, text: str) -> List[float]:
        dense, _sparse = embed_query_hybrid(text)
        return dense

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

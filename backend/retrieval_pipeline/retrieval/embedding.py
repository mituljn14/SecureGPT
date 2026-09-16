import logging
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None

log = logging.getLogger(__name__)

_model = None

def get_bge_m3_model():
    global _model
    if _model is None:
        if BGEM3FlagModel is None:
            raise ImportError("FlagEmbedding is not installed. Please run: pip install FlagEmbedding==1.3.3")
        
        log.info("Loading BGEM3FlagModel for hybrid retrieval...")
        _model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=torch.cuda.is_available())
    return _model

def embed_query_hybrid(text: str):
    """
    Returns (dense_vector: list, sparse_vector: dict)
    sparse_vector is of the form: {token_id_str: weight_float}
    """
    model = get_bge_m3_model()
    # Ensure it's passed as a list of strings for single inference
    out = model.encode([text], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    dense = out['dense_vecs'][0].tolist()
    sparse_dict = out['lexical_weights'][0]
    return dense, sparse_dict


def embed_text(text: str) -> list:
    """Dense-only embedding for one string.

    Convenience wrapper so callers that do not need the sparse half (the
    meeting agent's transcript indexer and query engine) still go through
    THIS embedder rather than standing up a second BGE-M3 instance. Two
    embedders in one process is both wasteful and unsafe: vectors that
    disagree about their space fail silently, returning plausible nonsense
    instead of an error.
    """
    dense, _sparse = embed_query_hybrid(text)
    return dense


def embed_texts(texts, batch_size: int = 16) -> list:
    """Dense embeddings for a list of strings, batched through one encode()
    call rather than one model round-trip per item."""
    if not texts:
        return []
    model = get_bge_m3_model()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        res = model.encode(chunk, return_dense=True, return_sparse=False,
                           return_colbert_vecs=False)
        out.extend(v.tolist() for v in res["dense_vecs"])
    return out

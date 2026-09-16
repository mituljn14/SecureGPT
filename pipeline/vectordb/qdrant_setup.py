#!/usr/bin/env python3
"""
Create the securegpt_v2 collection with hybrid (dense + sparse) vectors,
quantization, and the payload indexes that make RBAC and filtering both
correct and fast.

    python pipeline/vectordb/qdrant_setup.py            # create if absent
    python pipeline/vectordb/qdrant_setup.py --recreate # drop and rebuild

Key differences from the old securegpt_corpus collection:

  * access_roles / department are KEYWORD indexes, not TEXT. A text index
    is a tokenised substring match -- wrong semantics for an access
    control field, where only exact set membership is acceptable.
  * fiscal_year / fiscal_quarter are INTEGER indexes, so year filtering is
    an exact numeric comparison instead of MatchText against a filename
    (which also matched "20271" and "FY2027_v2").
  * named vectors: "dense" (1024-d cosine) and "sparse", both emitted by
    BGE-M3 in one forward pass, enabling server-side RRF hybrid search.
  * int8 scalar quantization with always_ram -- ~4x less memory for the
    dense index (763k x 1024 x 4B = 3.1GB -> ~780MB) with rescoring
    enabled so recall loss is negligible.
  * department carries is_tenant=True so Qdrant co-locates each
    department's points on disk, which speeds up the filtered searches
    that every RBAC query performs.
"""
import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

COLLECTION = os.getenv("QDRANT_COLLECTION", "securegpt_v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DENSE_DIM = 1024

# field -> index type
KEYWORD_FIELDS = [
    "access_roles", "sub_department", "chunk_type", "source_format",
    "sensitivity", "region", "doc_id", "parent_block_id", "content_hash",
    "source_path", "functional_area", "identifiers",
]
INTEGER_FIELDS = ["fiscal_year", "fiscal_quarter", "fiscal_half", "token_count", "chunk_index"]
TEXT_FIELDS = ["text", "file_name", "doc_title", "section_title"]


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=120)


def create(recreate: bool = False):
    client = get_client()

    if client.collection_exists(COLLECTION):
        if not recreate:
            info = client.get_collection(COLLECTION)
            print(f"Collection '{COLLECTION}' already exists ({info.points_count:,} points). "
                  f"Use --recreate to rebuild.")
            return client
        print(f"Deleting existing '{COLLECTION}' ...")
        client.delete_collection(COLLECTION)

    print(f"Creating '{COLLECTION}' (dense {DENSE_DIM}d cosine + sparse, int8 quantized) ...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": qm.VectorParams(
                size=DENSE_DIM,
                distance=qm.Distance.COSINE,
                quantization_config=qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False))
        },
        # payload lives on disk; only indexed fields stay hot in RAM
        on_disk_payload=True,
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=4),
    )

    # department is the primary partition key for every RBAC query
    print("  index: department (keyword, tenant)")
    client.create_payload_index(
        collection_name=COLLECTION, field_name="department",
        field_schema=qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True),
    )

    for field in KEYWORD_FIELDS:
        print(f"  index: {field} (keyword)")
        client.create_payload_index(
            collection_name=COLLECTION, field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    for field in INTEGER_FIELDS:
        print(f"  index: {field} (integer)")
        client.create_payload_index(
            collection_name=COLLECTION, field_name=field,
            field_schema=qm.PayloadSchemaType.INTEGER,
        )

    for field in TEXT_FIELDS:
        print(f"  index: {field} (full-text)")
        client.create_payload_index(
            collection_name=COLLECTION, field_name=field,
            field_schema=qm.TextIndexParams(
                type=qm.TextIndexType.TEXT,
                tokenizer=qm.TokenizerType.WORD,
                min_token_len=2, max_token_len=30, lowercase=True,
            ),
        )

    info = client.get_collection(COLLECTION)
    print(f"\nCreated '{COLLECTION}'. status={info.status}  points={info.points_count}")
    return client


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()
    create(recreate=args.recreate)

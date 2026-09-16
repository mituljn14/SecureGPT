"""
Test Qdrant database connection, collection discovery, and non-destructive retrieval.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.retrieval.qdrant_client import get_qdrant_wrapper
from vision_agent.retrieval.retriever import KnowledgeRetriever
from vision_agent.config.settings import settings


def test_qdrant_listing():
    qdrant = get_qdrant_wrapper()
    assert qdrant.check_connection(), "Qdrant unreachable"
    collections = qdrant.list_collections()
    assert len(collections) > 0, "No collections found"
    col_names = [c["name"] for c in collections]
    print(f"[OK] Available collections: {col_names}")
    assert settings.QDRANT_COLLECTION in col_names, f"Target collection '{settings.QDRANT_COLLECTION}' not in Qdrant"


def test_qdrant_retrieval():
    retriever = KnowledgeRetriever()
    results = retriever.retrieve("financial report revenue", role="CEO", top_k=2)
    print(f"[OK] Retrieved {len(results)} chunk(s) from Qdrant")
    for r in results:
        print(f"   [Doc: {r.file_name} | Score: {r.score:.3f}] {r.text[:80]}...")


if __name__ == "__main__":
    test_qdrant_listing()
    test_qdrant_retrieval()
    print("Qdrant tests passed successfully.")

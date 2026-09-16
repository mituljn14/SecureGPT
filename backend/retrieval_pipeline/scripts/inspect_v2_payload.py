import os
import sys
# Add backend to path so we can import from retrieval_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from retrieval_pipeline.retrieval.qdrant_client import get_client
COLLECTION_NAME = "securegpt_v2"

client = get_client()

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=200,
    with_payload=True,
    with_vectors=False,
)

for p in points:
    file_name = p.payload.get("file_name", "")
    if file_name.endswith(".pdf"):
        print(f"File: {file_name}")
        print(f"Role: {p.payload.get('access_roles')}")
        print(f"Text snippet: {p.payload.get('text', '')[:200]}")
        print("-" * 50)

from quadrant_embeddings.vectordb.qdrant_store import get_client, COLLECTION_NAME

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

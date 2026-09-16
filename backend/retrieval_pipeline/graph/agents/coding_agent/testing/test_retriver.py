"""
test_retriever.py

STEP 1 — Minimal sanity test for the Retriever.

This is a real integration test (not a mock) — it requires:
    - Qdrant running at http://localhost:6333 with `securegpt_corpus` populated
    - Ollama running at http://localhost:11434 with `bge-m3` pulled

Run:
    python test_retriever.py
"""

from retriever import Retriever


def test_basic_search():
    retriever = Retriever()
    results = retriever.search("average sales for 2025", top_k=3)

    assert isinstance(results, list), "search() should return a list"
    assert len(results) > 0, "Expected at least one result from a populated collection"

    first = results[0]
    required_fields = {
        "chunk_id", "document_id", "file_name",
        "source_hash", "chunk_index", "text", "score",
    }
    missing = required_fields - set(first.keys())
    assert not missing, f"Result is missing fields: {missing}"

    assert isinstance(first["score"], float), "score should be a float"

    print("PASS: basic_search")
    print(f"  Top result document_id={first['document_id']} score={first['score']:.4f}")


def test_metadata_filter():
    retriever = Retriever()

    # First, find a real file_name to filter on
    baseline = retriever.search("sales report", top_k=1)
    if not baseline:
        print("SKIP: metadata_filter (no baseline result to derive a file_name from)")
        return

    file_name = baseline[0]["file_name"]
    if not file_name:
        print("SKIP: metadata_filter (baseline result had no file_name payload)")
        return

    filtered = retriever.search(
        "sales report", top_k=3, metadata_filters={"file_name": file_name}
    )
    assert all(r["file_name"] == file_name for r in filtered), \
        "All filtered results should match the requested file_name"

    print(f"PASS: metadata_filter (file_name={file_name})")


if __name__ == "__main__":
    test_basic_search()
    test_metadata_filter()
    print("\nAll Step 1 tests passed.")
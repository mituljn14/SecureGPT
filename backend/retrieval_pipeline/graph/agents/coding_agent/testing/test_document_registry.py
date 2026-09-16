"""
test_document_registry.py

STEP 2 — Minimal sanity test for the Document Registry.

Uses real document_id values that came back from the Step 1 Retriever
test run, so this actually exercises production data, not fixtures.

Run:
    python test_document_registry.py
"""

from document_registry import DocumentRegistry

# Pulled straight from your Step 1 retriever.py output.
KNOWN_DOCUMENT_IDS = ["DOC-002841", "DOC-000164", "DOC-000159", "DOC-002810"]


def test_single_lookup():
    registry = DocumentRegistry()
    result = registry.get_by_document_id("DOC-002841")

    assert result is not None, "Expected DOC-002841 to exist in the registry"
    assert result["file_name"] == "Budget_Tracker_Q3_2025.xlsx", \
        f"file_name mismatch: {result['file_name']}"
    assert result["source_hash"] == "6caae9126c5e1396", \
        f"source_hash mismatch: {result['source_hash']}"
    assert result["source_path"], "source_path should not be empty"

    print("PASS: single_lookup")
    print(f"  document_id={result['document_id']}")
    print(f"  source_path={result['source_path']}")

    registry.close()


def test_batch_lookup():
    registry = DocumentRegistry()
    results = registry.get_many(KNOWN_DOCUMENT_IDS)

    missing = set(KNOWN_DOCUMENT_IDS) - set(results.keys())
    assert not missing, f"Expected all known IDs to resolve, missing: {missing}"

    for doc_id, record in results.items():
        assert record["source_path"], f"{doc_id} has empty source_path"

    print(f"PASS: batch_lookup ({len(results)}/{len(KNOWN_DOCUMENT_IDS)} resolved)")

    registry.close()


def test_missing_id():
    registry = DocumentRegistry()
    result = registry.get_by_document_id("DOC-999999-DOES-NOT-EXIST")
    assert result is None, "Expected None for a nonexistent document_id"
    print("PASS: missing_id returns None as expected")

    registry.close()


def test_hash_verification():
    registry = DocumentRegistry()
    correct = registry.verify_source_hash("DOC-002841", "6caae9126c5e1396")
    wrong = registry.verify_source_hash("DOC-002841", "not_the_real_hash")

    assert correct is True, "Correct hash should verify as True"
    assert wrong is False, "Wrong hash should verify as False"

    print("PASS: hash_verification")

    registry.close()


if __name__ == "__main__":
    test_single_lookup()
    test_batch_lookup()
    test_missing_id()
    test_hash_verification()
    print("\nAll Step 2 tests passed.")
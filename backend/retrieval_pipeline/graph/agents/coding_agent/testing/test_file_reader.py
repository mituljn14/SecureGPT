"""
test_file_reader.py

STEP 3 — Sanity test for the File Reader.

This chains Step 2 -> Step 3 for real: it looks up DOC-002841's
source_path in document_registry.db, then actually reads that file
from disk. This requires document_registry.db and the real data/
folder to be reachable, same as your Step 1/2 runs.

Run:
    python test_file_reader.py
"""

from document_registry import DocumentRegistry
from file_reader import FileReader


def test_inspect_xlsx_via_registry():
    registry = DocumentRegistry()
    record = registry.get_by_document_id("DOC-002841")
    registry.close()

    assert record is not None, "DOC-002841 should exist in the registry"
    source_path = record["source_path"]
    file_type = record["file_type"]

    reader = FileReader()
    result = reader.inspect(source_path, file_type=file_type)

    assert result["type"] == "tabular"
    assert result["file_type"] == "xlsx"
    assert len(result["sheets"]) > 0, "Expected at least one sheet"

    first_sheet_name = list(result["sheets"].keys())[0]
    first_sheet = result["sheets"][first_sheet_name]
    assert "columns" in first_sheet
    assert "num_rows" in first_sheet
    assert len(first_sheet["preview"]) > 0, "Expected non-empty preview rows"

    print("PASS: inspect_xlsx_via_registry")
    print(f"  source_path={source_path}")
    print(f"  sheet='{first_sheet_name}' columns={first_sheet['columns']}")
    print(f"  num_rows={first_sheet['num_rows']}")


def test_read_full_xlsx_via_registry():
    registry = DocumentRegistry()
    record = registry.get_by_document_id("DOC-002841")
    registry.close()

    reader = FileReader()
    data = reader.read_full(record["source_path"], file_type=record["file_type"])

    assert isinstance(data, dict), "xlsx read_full should return dict of sheet DataFrames"
    for sheet_name, df in data.items():
        assert hasattr(df, "columns"), f"Sheet '{sheet_name}' should be a DataFrame"

    print(f"PASS: read_full_xlsx_via_registry ({len(data)} sheet(s) loaded)")


if __name__ == "__main__":
    test_inspect_xlsx_via_registry()
    test_read_full_xlsx_via_registry()
    print("\nAll Step 3 tests passed.")
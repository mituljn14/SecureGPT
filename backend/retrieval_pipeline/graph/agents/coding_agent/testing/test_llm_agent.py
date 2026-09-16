"""
test_llm_agent.py

Small standalone test for Step 4 (llm_agent.py).

This test only requires Ollama to be running locally with qwen2.5-coder:3b
pulled. It does NOT touch Qdrant, document_registry.db, or the real data/
folder — it uses a fixed fake file_context so it can run before Steps 1-3
are wired together.

Run:
    python test_llm_agent.py
"""

import sys
from llm_agent import generate_code, _extract_code, build_user_prompt


def test_extract_code_with_fences():
    raw = "Here is code:\n```python\nprint('hi')\n```\n"
    code = _extract_code(raw)
    assert code == "print('hi')", f"Unexpected extraction: {code!r}"
    print("PASS: extract_code_with_fences")


def test_extract_code_no_fences():
    raw = "print('no fences here')"
    code = _extract_code(raw)
    assert code == "print('no fences here')"
    print("PASS: extract_code_no_fences")


def test_build_user_prompt_contains_query_and_path():
    ctx = {"type": "tabular", "source_path": "data/foo.xlsx", "sheets": {}}
    prompt = build_user_prompt("Calculate average sales", ctx)
    assert "Calculate average sales" in prompt
    assert "data/foo.xlsx" in prompt
    print("PASS: build_user_prompt_contains_query_and_path")


def test_generate_code_live():
    """Requires a running Ollama server with qwen2.5-coder:3b pulled."""
    file_context = {
        "type": "tabular",
        "file_type": "xlsx",
        "source_path": "data/Executive_Corporate/Corporate_Strategy/Business_Planning/Budget_Tracker_Q3_2025.xlsx",
        "sheets": {
            "Budget_Tracker": {
                "columns": ["Cost_Center", "Category", "Month",
                            "Budgeted_INR_Lakh", "Actual_INR_Lakh", "Variance_%"],
                "dtypes": {
                    "Cost_Center": "object",
                    "Category": "object",
                    "Month": "object",
                    "Budgeted_INR_Lakh": "float64",
                    "Actual_INR_Lakh": "float64",
                    "Variance_%": "float64",
                },
                "num_rows": 108,
                "preview": [
                    {"Cost_Center": "CC-101", "Category": "R&D", "Month": "Jan-2025",
                     "Budgeted_INR_Lakh": 12.5, "Actual_INR_Lakh": 11.9, "Variance_%": -4.8},
                ],
            }
        },
    }

    try:
        result = generate_code("Calculate the average Actual_INR_Lakh grouped by Category",
                                file_context)
    except ConnectionError as e:
        print(f"SKIP: test_generate_code_live (Ollama not reachable: {e})")
        return

    assert "code" in result and len(result["code"]) > 0, "No code generated"
    assert "pandas" in result["code"] or "pd." in result["code"], \
        "Generated code doesn't appear to use pandas"
    print("PASS: generate_code_live")
    print("----- Generated code -----")
    print(result["code"])
    print("---------------------------")


if __name__ == "__main__":
    test_extract_code_with_fences()
    test_extract_code_no_fences()
    test_build_user_prompt_contains_query_and_path()
    test_generate_code_live()
    print("\nAll Step 4 tests completed.")
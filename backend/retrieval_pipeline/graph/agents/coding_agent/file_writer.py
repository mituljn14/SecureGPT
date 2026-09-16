"""
file_writer.py
Step 7: File Writer

Writes agent output to disk ONLY when the user explicitly requested a file
(e.g. "save this as xlsx", "export to csv"). Never called otherwise.

Supported formats: csv, xlsx, docx, json, txt
All outputs are restricted to OUTPUT_DIR (no path traversal).
"""

import os
import json
import re
import pandas as pd
from docx import Document


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "docx", "json", "txt"}


class FileWriter:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _safe_filename(self, filename: str, ext: str) -> str:
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported output extension: {ext}")

        raw = filename.strip()
        # Reject traversal / path-separator attempts explicitly, BEFORE sanitizing.
        # (Sanitizing first would silently strip "../" down to "evil" and mask the
        #  attack instead of rejecting it.)
        if not raw:
            raise ValueError("Filename cannot be empty.")
        if ".." in raw or "/" in raw or "\\" in raw or raw.startswith("~"):
            raise ValueError(f"Invalid filename (path traversal not allowed): {filename!r}")

        name = re.sub(r"[^A-Za-z0-9_\-]", "_", raw)
        if not name:
            raise ValueError(f"Filename resolves to empty after sanitization: {filename!r}")
        return f"{name}.{ext}"

    def _resolve(self, filename: str, ext: str) -> str:
        safe_name = self._safe_filename(filename, ext)
        full_path = os.path.join(self.output_dir, safe_name)
        # Defense-in-depth backstop, redundant with the check above but kept in case
        # this method is ever called with a pre-sanitized name from elsewhere.
        if os.path.commonpath([os.path.abspath(full_path), os.path.abspath(self.output_dir)]) != os.path.abspath(self.output_dir):
            raise ValueError("Invalid output path.")
        return full_path

    def write_csv(self, filename: str, rows: list, columns: list = None) -> str:
        path = self._resolve(filename, "csv")
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return path

    def write_xlsx(self, filename: str, sheets: dict) -> str:
        """sheets = {sheet_name: [ {col: val, ...}, ... ]}"""
        path = self._resolve(filename, "xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, rows in sheets.items():
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=(sheet_name or "Sheet1")[:31], index=False)
        return path

    def write_docx(self, filename: str, title: str, paragraphs: list) -> str:
        path = self._resolve(filename, "docx")
        doc = Document()
        if title:
            doc.add_heading(title, level=1)
        for p in paragraphs:
            doc.add_paragraph(p)
        doc.save(path)
        return path

    def write_json(self, filename: str, data) -> str:
        path = self._resolve(filename, "json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def write_txt(self, filename: str, text: str) -> str:
        path = self._resolve(filename, "txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


if __name__ == "__main__":
    fw = FileWriter()

    p1 = fw.write_csv("test_sales", [{"Month": "Jan", "Sales": 100}, {"Month": "Feb", "Sales": 150}])
    print("PASS: write_csv ->", p1); assert os.path.exists(p1)

    p2 = fw.write_xlsx("test_report", {"Summary": [{"Metric": "Average Sales", "Value": 4523.7}]})
    print("PASS: write_xlsx ->", p2); assert os.path.exists(p2)

    p3 = fw.write_docx("test_doc", "Sales Report", ["Average sales for 2025 was 4523.7 INR Lakh."])
    print("PASS: write_docx ->", p3); assert os.path.exists(p3)

    p4 = fw.write_json("test_result", {"answer": 4523.7})
    print("PASS: write_json ->", p4); assert os.path.exists(p4)

    try:
        fw._resolve("../../evil", "csv")
        raise AssertionError("Path traversal was not blocked!")
    except ValueError:
        print("PASS: path traversal blocked")

    print("\nAll Step 7 tests passed.")
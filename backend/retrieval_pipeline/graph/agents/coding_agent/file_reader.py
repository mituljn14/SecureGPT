"""
file_reader.py

STEP 3 — File Reader ONLY.

This module does exactly one job:
    Given a source_path (as returned by the Document Registry in
    Step 2) and its file_type, read the ORIGINAL file and extract
    structured data from it.

Two entry points:
    inspect(source_path)   -> cheap schema + small preview.
                               This is what gets handed to Qwen so it
                               can write correct pandas/PyMuPDF/etc
                               code without needing the whole file
                               stuffed into its prompt.

    read_full(source_path) -> the actual full data (DataFrame(s) for
                               tabular files, full text for PDF/DOCX).
                               Used for verification or small files,
                               not for building Qwen's prompt.

It does NOT:
    - call Qwen
    - run any sandboxed code
    - write anything

Requirements:
    pip install pandas openpyxl pymupdf python-docx

Config:
    If your document_registry.db stores source_path as relative paths
    (relative to the `data/` folder), set DATA_ROOT below. If it
    stores absolute paths, leave DATA_ROOT = "" and it will be ignored.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import pymupdf as fitz  # PyMuPDF (pymupdf import name; fitz is the deprecated alias)
import docx  # python-docx


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# v2 payloads store `source_path` RELATIVE to the corpus root
# (e.g. "Operations/COO/Production_Reports/Work_Order_Data_Q3_2022.csv"),
# so an empty DATA_ROOT would fail to resolve every single one. CORPUS_ROOT
# is the same value the rest of the pipeline uses -- see
# retrieval/config.py -- rather than a second, drifting copy.
from retrieval_pipeline.retrieval.config import CORPUS_ROOT

DATA_ROOT = str(CORPUS_ROOT)

# Cap how many rows go into inspect()'s preview — keep Qwen's prompt small.
PREVIEW_ROWS = 5
# Cap how much text goes into inspect()'s preview for PDF/DOCX.
PREVIEW_CHARS = 1000


class FileReader:
    def __init__(self, data_root: str = DATA_ROOT):
        self.data_root = Path(data_root) if data_root else None

    # ----------------------------------------------------------------
    def _resolve_path(self, source_path: str) -> Path:
        path = Path(source_path)
        if path.exists():
            return path
        if self.data_root:
            candidate = self.data_root / source_path
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find file at '{source_path}'"
            + (f" or '{self.data_root / source_path}'" if self.data_root else "")
            + ". Check DATA_ROOT in file_reader.py, or that source_path in "
              "document_registry.db is correct."
        )

    @staticmethod
    def _file_type_from_path(path: Path) -> str:
        return path.suffix.lower().lstrip(".")

    # ==================================================================
    # inspect() — cheap schema + preview
    # ==================================================================
    def inspect(self, source_path: str, file_type: Optional[str] = None) -> dict:
        path = self._resolve_path(source_path)
        ftype = (file_type or self._file_type_from_path(path)).lower()

        if ftype == "csv":
            return self._inspect_csv(path)
        elif ftype in ("xlsx", "xls"):
            return self._inspect_xlsx(path)
        elif ftype == "pdf":
            return self._inspect_pdf(path)
        elif ftype == "docx":
            return self._inspect_docx(path)
        elif ftype == "pptx":
            return self._inspect_pptx(path)
        elif ftype in ("txt", "md", "log", "json", "tsv"):
            return self._inspect_text(path)
        else:
            raise ValueError(f"Unsupported file_type: {ftype}")

    def _inspect_pptx(self, path: Path) -> dict:
        """Slide-by-slide text, plus any tables found on the slides."""
        from pptx import Presentation

        prs = Presentation(str(path))
        slides, tables = [], []
        for idx, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    texts.append(shape.text_frame.text.strip())
                if getattr(shape, "has_table", False):
                    rows = [[c.text for c in r.cells] for r in shape.table.rows]
                    if rows:
                        tables.append({"slide": idx, "rows": rows[:20]})
            slides.append({"slide": idx, "text": "\n".join(texts)[:PREVIEW_CHARS]})
        return {
            "type": "presentation",
            "file_type": "pptx",
            "source_path": path.as_posix(),
            "num_slides": len(slides),
            "slides": slides[:15],
            "tables": tables[:10],
        }

    def _inspect_text(self, path: Path) -> dict:
        """Plain-text preview with a line count."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "type": "text",
            "file_type": path.suffix.lstrip(".").lower(),
            "source_path": path.as_posix(),
            "num_lines": text.count("\n") + 1,
            "num_chars": len(text),
            "preview": text[:PREVIEW_CHARS],
        }

    def _inspect_csv(self, path: Path) -> dict:
        df = pd.read_csv(path)
        return {
            "type": "tabular",
            "file_type": "csv",
            "source_path": path.as_posix(),
            "sheets": {
                "default": {
                    "columns": list(df.columns),
                    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                    "num_rows": len(df),
                    "preview": df.head(PREVIEW_ROWS).to_dict(orient="records"),
                }
            },
        }

    def _inspect_xlsx(self, path: Path) -> dict:
        sheets_dict = pd.read_excel(path, sheet_name=None)  # all sheets
        sheets = {}
        for name, df in sheets_dict.items():
            sheets[name] = {
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "num_rows": len(df),
                "preview": df.head(PREVIEW_ROWS).to_dict(orient="records"),
            }
        return {
            "type": "tabular",
            "file_type": "xlsx",
            "source_path": path.as_posix(),
            "sheets": sheets,
        }

    def _inspect_pdf(self, path: Path) -> dict:
        doc = fitz.open(path)
        num_pages = doc.page_count
        preview_text = ""
        for page in doc:
            preview_text += page.get_text()
            if len(preview_text) >= PREVIEW_CHARS:
                break
        doc.close()
        return {
            "type": "text",
            "file_type": "pdf",
            "source_path": path.as_posix(),
            "num_pages": num_pages,
            "preview": preview_text[:PREVIEW_CHARS],
        }

    def _inspect_docx(self, path: Path) -> dict:
        document = docx.Document(path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        preview_text = "\n".join(paragraphs)[:PREVIEW_CHARS]
        return {
            "type": "text",
            "file_type": "docx",
            "source_path": path.as_posix(),
            "num_paragraphs": len(paragraphs),
            "num_tables": len(document.tables),
            "preview": preview_text,
        }

    # ==================================================================
    # read_full() — actual full data
    # ==================================================================
    def read_full(self, source_path: str, file_type: Optional[str] = None):
        path = self._resolve_path(source_path)
        ftype = (file_type or self._file_type_from_path(path)).lower()

        if ftype == "csv":
            return pd.read_csv(path)
        elif ftype in ("xlsx", "xls"):
            return pd.read_excel(path, sheet_name=None)  # dict of {sheet_name: DataFrame}
        elif ftype == "pdf":
            doc = fitz.open(path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        elif ftype == "docx":
            document = docx.Document(path)
            paragraphs = [p.text for p in document.paragraphs]
            tables = []
            for table in document.tables:
                tables.append(
                    [[cell.text for cell in row.cells] for row in table.rows]
                )
            return {"paragraphs": paragraphs, "tables": tables}
        else:
            raise ValueError(f"Unsupported file_type: {ftype}")


# --------------------------------------------------------------------------
# Standalone test entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python file_reader.py <source_path> [file_type]")
        sys.exit(1)

    src = sys.argv[1]
    ftype = sys.argv[2] if len(sys.argv) > 2 else None

    reader = FileReader()
    result = reader.inspect(src, file_type=ftype)
    print(json.dumps(result, indent=2, default=str))
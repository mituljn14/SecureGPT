# pipeline/converters/pdf_to_md.py
"""
PDF -> Markdown via pymupdf4llm.

Born-digital PDFs (not scans) -> pymupdf4llm's layout analysis produces
real markdown tables (header + separator row) and heading detection from
font size, which is exactly what we validated on Annual_Report_2018.pdf.
CPU-only, no OCR/vision model needed.

Fixes applied on top of the raw pymupdf4llm output:
  - ftfy.fix_text() to repair mojibake (e.g. a mis-decoded en-dash showing
    up as "5-Year Comparison" -> garbled bytes, observed in the original
    test_pymupdf_output.md).
"""
import ftfy
import pymupdf4llm


def convert(file_path: str) -> str:
    md = pymupdf4llm.to_markdown(file_path)
    return ftfy.fix_text(md)

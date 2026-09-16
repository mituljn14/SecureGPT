"""
Test PDF processing, page rendering, and text extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.document.pdf_processor import PDFProcessor
from vision_agent.config.settings import settings


def create_synthetic_pdf(target_path: Path) -> Path:
    """Generates a small multi-page test PDF with text and shapes."""
    doc = pymupdf.open()
    # Page 1: Native Text
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 72), "SecureGPT Vision Agent Report\nQuarterly Project Metrics", fontsize=16)
    p1.insert_text((50, 120), "Page 1 contains native selectable text for extraction testing.", fontsize=12)

    # Page 2: Native table
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 72), "Financial Summary", fontsize=14)
    p2.draw_rect(pymupdf.Rect(50, 100, 500, 200), color=(0, 0, 0), width=1)
    p2.insert_text((60, 130), "Metric | Q1 2026 | Q2 2026", fontsize=11)
    p2.insert_text((60, 160), "Revenue | $120,000 | $150,000", fontsize=11)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(target_path))
    doc.close()
    return target_path


def test_pdf_processing():
    test_pdf_path = settings.TEMP_DIR / "test_sample.pdf"
    create_synthetic_pdf(test_pdf_path)

    proc = PDFProcessor()
    pages = proc.process_pdf(test_pdf_path)

    print(f"[OK] PDF Processed {len(pages)} page(s)")
    assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"
    assert pages[0].has_selectable_text, "Page 1 should have selectable text"
    assert "SecureGPT" in pages[0].native_text
    print(f"[OK] Page 1 Text: {pages[0].native_text[:40]}...")


if __name__ == "__main__":
    test_pdf_processing()
    print("PDF tests passed successfully.")

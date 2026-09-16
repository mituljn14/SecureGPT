"""
Extracts native selectable text from PDFs and text documents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import pymupdf

from vision_agent.utils.logging_utils import get_logger

logger = get_logger("text_extractor")


class TextExtractor:
    @staticmethod
    def extract_from_pdf(pdf_path: Path) -> List[Dict[str, str]]:
        """
        Extract selectable text page by page from a PDF file.
        Returns a list of dicts: [{"page": 1, "text": "...", "has_text": True/False}, ...]
        """
        results: List[Dict[str, str]] = []
        try:
            doc = pymupdf.open(str(pdf_path))
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text").strip()
                results.append(
                    {
                        "page": page_index + 1,
                        "text": text,
                        "has_text": len(text) > 30,  # meaningful text threshold
                    }
                )
            doc.close()
        except Exception as exc:
            logger.error(f"Error extracting text from PDF {pdf_path}: {exc}")
            raise
        return results

    @staticmethod
    def extract_from_text_file(file_path: Path) -> str:
        """Reads plain text / markdown / CSV, and Office documents.

        PowerPoint and Word route here (file_utils.OFFICE_EXTENSIONS) because
        reading them as text is what lets the pipeline summarise and ground
        them -- a raw read_text() on a .pptx returns zip bytes.
        """
        suffix = file_path.suffix.lower()
        try:
            if suffix in (".pptx", ".ppt"):
                from pptx import Presentation
                prs = Presentation(str(file_path))
                out = []
                for idx, slide in enumerate(prs.slides, start=1):
                    body = [
                        shape.text_frame.text.strip()
                        for shape in slide.shapes
                        if shape.has_text_frame and shape.text_frame.text.strip()
                    ]
                    out.append(f"--- Slide {idx} ---\n" + "\n".join(body))
                return "\n\n".join(out)

            if suffix in (".docx", ".doc"):
                import docx
                d = docx.Document(str(file_path))
                paras = [p.text for p in d.paragraphs if p.text.strip()]
                for table in d.tables:
                    for row in table.rows:
                        cells = [c.text.strip() for c in row.cells]
                        if any(cells):
                            paras.append(" | ".join(cells))
                return "\n".join(paras)

            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.error(f"Failed to read text file {file_path}: {exc}")
            raise

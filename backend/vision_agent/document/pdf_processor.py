"""
Comprehensive PDF processor for text, scanned pages, mixed content, and rendering.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pymupdf
from PIL import Image
import io

from vision_agent.core.models import ProcessedPage
from vision_agent.document.text_extractor import TextExtractor
from vision_agent.document.ocr_processor import OCRProcessor
from vision_agent.document.table_extractor import TableExtractor
from vision_agent.utils.image_utils import save_temp_image
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("pdf_processor")


class PDFProcessor:
    def __init__(
        self,
        ocr_processor: Optional[OCRProcessor] = None,
        render_dpi: int = 150,
    ):
        self.ocr = ocr_processor or OCRProcessor()
        self.render_dpi = render_dpi

    def process_pdf(
        self,
        pdf_path: Path,
        force_ocr: bool = False,
        render_all_pages: bool = True,
    ) -> List[ProcessedPage]:
        """
        Processes PDF page by page, rendering images, extracting native text,
        running OCR when needed, and extracting tables.
        """
        processed_pages: List[ProcessedPage] = []
        doc = pymupdf.open(str(pdf_path))
        total_pages = len(doc)
        logger.info(f"Processing PDF '{pdf_path.name}' with {total_pages} page(s)...")

        # Extract native tables
        native_tables = TableExtractor.extract_tables_from_pdf(pdf_path)
        tables_by_page = {}
        for tbl in native_tables:
            tables_by_page.setdefault(tbl.page_number, []).append(tbl)

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Native text extraction
            native_text = page.get_text("text").strip()
            has_selectable_text = len(native_text) >= 40

            # 2. Render page to image
            rendered_image_path = None
            pil_image = None
            if render_all_pages or not has_selectable_text or force_ocr:
                pix = page.get_pixmap(dpi=self.render_dpi)
                pil_image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                rendered_image_path = save_temp_image(pil_image, prefix=f"page_{page_num}")

            # 3. OCR execution if scanned or text is sparse
            ocr_result = None
            if (not has_selectable_text or force_ocr) and pil_image is not None:
                logger.debug(f"Running OCR on scanned/image page {page_num}...")
                ocr_result = self.ocr.process_image(pil_image, page_number=page_num)

            # 4. Check for visual elements (images, vector drawings)
            images_list = page.get_images()
            drawings = page.get_drawings()
            has_visuals = len(images_list) > 0 or len(drawings) > 5

            processed_pages.append(
                ProcessedPage(
                    page_number=page_num,
                    image_path=rendered_image_path,
                    has_selectable_text=has_selectable_text,
                    native_text=native_text,
                    ocr_result=ocr_result,
                    tables=tables_by_page.get(page_num, []),
                    chart_detected=has_visuals,
                    diagram_detected=has_visuals and len(drawings) > 10,
                )
            )

        doc.close()
        return processed_pages

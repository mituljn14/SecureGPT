"""
Unified Document Analyzer for ingesting PDFs, standalone images, spreadsheets, and text.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from PIL import Image

from vision_agent.core.models import ProcessedPage, TableData
from vision_agent.document.pdf_processor import PDFProcessor
from vision_agent.document.ocr_processor import OCRProcessor
from vision_agent.document.table_extractor import TableExtractor
from vision_agent.utils.file_utils import detect_file_type
from vision_agent.utils.image_utils import load_image_safely, save_temp_image
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("document_analyzer")


class DocumentAnalyzer:
    def __init__(
        self,
        pdf_processor: PDFProcessor | None = None,
        ocr_processor: OCRProcessor | None = None,
    ):
        self.ocr = ocr_processor or OCRProcessor()
        self.pdf_proc = pdf_processor or PDFProcessor(ocr_processor=self.ocr)

    def analyze(
        self,
        file_path: Path,
        force_ocr: bool = False,
    ) -> Tuple[str, List[ProcessedPage], List[TableData]]:
        """
        Analyzes any supported document or image.
        Returns (detected_type, pages_list, tables_list).
        """
        detected_type = detect_file_type(file_path)
        pages: List[ProcessedPage] = []
        tables: List[TableData] = []

        if detected_type == "pdf":
            pages = self.pdf_proc.process_pdf(file_path, force_ocr=force_ocr)
            for p in pages:
                tables.extend(p.tables)

        elif detected_type == "image":
            pil_img = load_image_safely(file_path)
            temp_path = save_temp_image(pil_img, prefix="img")
            ocr_res = self.ocr.process_image(pil_img, page_number=1)

            # Try to parse markdown table from OCR if present
            table_obj = TableExtractor.parse_markdown_table(ocr_res.full_text)
            page_tables = [table_obj] if table_obj else []
            tables.extend(page_tables)

            pages = [
                ProcessedPage(
                    page_number=1,
                    image_path=temp_path,
                    has_selectable_text=False,
                    native_text="",
                    ocr_result=ocr_res,
                    tables=page_tables,
                    chart_detected=True,
                    diagram_detected=True,
                )
            ]

        elif detected_type == "table_data":
            extracted_tables = TableExtractor.extract_from_data_file(file_path)
            tables.extend(extracted_tables)
            for idx, tbl in enumerate(extracted_tables):
                pages.append(
                    ProcessedPage(
                        page_number=idx + 1,
                        image_path=None,
                        has_selectable_text=True,
                        native_text=tbl.to_markdown(),
                        ocr_result=None,
                        tables=[tbl],
                    )
                )

        elif detected_type == "text":
            text_content = file_path.read_text(encoding="utf-8", errors="replace")
            tbl = TableExtractor.parse_markdown_table(text_content)
            if tbl:
                tables.append(tbl)
            pages = [
                ProcessedPage(
                    page_number=1,
                    image_path=None,
                    has_selectable_text=True,
                    native_text=text_content,
                    ocr_result=None,
                    tables=[tbl] if tbl else [],
                )
            ]

        else:
            raise ValueError(
                f"Unsupported file format '{file_path.suffix}'. "
                f"Supported: .pdf, .png, .jpg, .jpeg, .webp, .bmp, .tiff, .csv, .xlsx, .txt, .md"
            )

        return detected_type, pages, tables

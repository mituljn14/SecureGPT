"""
Independent OCR Processor using RapidOCR (ONNX) with pytesseract fallback.
Extracts text and bounding boxes while preserving layout and page information.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional
import numpy as np
from PIL import Image

from vision_agent.core.models import OCRResult, OCRBoundingBox
from vision_agent.utils.image_utils import load_image_safely
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("ocr_processor")


class OCRProcessor:
    def __init__(self, preferred_engine: str = "rapidocr"):
        self.preferred_engine = preferred_engine
        self._rapid_ocr = None
        self._rapid_ocr_init_failed = False
        self._pytesseract_available = None

    def _get_rapid_ocr(self):
        if self._rapid_ocr is None and not self._rapid_ocr_init_failed:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._rapid_ocr = RapidOCR()
                logger.debug("RapidOCR engine initialized successfully.")
            except Exception as exc:
                logger.warning(f"Could not initialize RapidOCR: {exc}")
                self._rapid_ocr_init_failed = True
        return self._rapid_ocr

    def _check_pytesseract(self) -> bool:
        if self._pytesseract_available is None:
            try:
                import pytesseract
                import shutil

                has_bin = shutil.which("tesseract") is not None
                self._pytesseract_available = has_bin
            except Exception:
                self._pytesseract_available = False
        return self._pytesseract_available

    def process_image(
        self,
        source: Union[str, Path, bytes, Image.Image],
        page_number: int = 1,
    ) -> OCRResult:
        """
        Perform OCR on an image or page, returning structured OCRResult.
        """
        try:
            pil_img = load_image_safely(source)
        except Exception as exc:
            return OCRResult(
                page_number=page_number,
                full_text="",
                boxes=[],
                success=False,
                error_message=f"Failed to load image for OCR: {exc}",
            )

        # 1. Try RapidOCR
        rapid_engine = self._get_rapid_ocr()
        if rapid_engine is not None:
            try:
                img_np = np.array(pil_img)
                ocr_result, _ = rapid_engine(img_np)
                if ocr_result:
                    boxes: list[OCRBoundingBox] = []
                    lines: list[str] = []
                    for item in ocr_result:
                        # item format: [box_points, text_str, confidence_str]
                        box_points = item[0]
                        text_str = str(item[1]).strip()
                        confidence = float(item[2]) if len(item) > 2 else 1.0
                        if text_str:
                            boxes.append(
                                OCRBoundingBox(
                                    box=box_points,
                                    text=text_str,
                                    confidence=confidence,
                                )
                            )
                            lines.append(text_str)

                    full_text = "\n".join(lines)
                    return OCRResult(
                        page_number=page_number,
                        full_text=full_text,
                        boxes=boxes,
                        engine_used="rapidocr",
                        success=True,
                    )
                else:
                    return OCRResult(
                        page_number=page_number,
                        full_text="",
                        boxes=[],
                        engine_used="rapidocr",
                        success=True,
                    )
            except Exception as exc:
                logger.warning(f"RapidOCR failed on page {page_number}: {exc}. Trying fallback...")

        # 2. Try pytesseract fallback if available
        if self._check_pytesseract():
            try:
                import pytesseract

                text = pytesseract.image_to_string(pil_img)
                return OCRResult(
                    page_number=page_number,
                    full_text=text.strip(),
                    boxes=[],
                    engine_used="pytesseract",
                    success=True,
                )
            except Exception as exc:
                logger.warning(f"pytesseract failed: {exc}")

        # If both fail or no text found
        return OCRResult(
            page_number=page_number,
            full_text="",
            boxes=[],
            engine_used="none",
            success=False,
            error_message="No suitable OCR engine could extract text.",
        )

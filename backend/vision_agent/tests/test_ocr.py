"""
Test OCR text extraction on synthetic images.
"""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.document.ocr_processor import OCRProcessor


def test_ocr_extraction():
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 35), "INVOICE TOTAL $1500", fill=(0, 0, 0))

    processor = OCRProcessor()
    result = processor.process_image(img, page_number=1)

    print(f"[OK] OCR Engine Used: {result.engine_used}")
    print(f"[OK] OCR Extracted Text: {result.full_text}")
    assert result.success, "OCR failed"
    assert "TOTAL" in result.full_text.upper() or "1500" in result.full_text or "INVOICE" in result.full_text


if __name__ == "__main__":
    test_ocr_extraction()
    print("OCR tests passed successfully.")

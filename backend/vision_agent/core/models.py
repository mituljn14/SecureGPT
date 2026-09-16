"""
Core data models and type definitions for the Vision Agent pipeline.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
import pandas as pd


class TaskType(str, Enum):
    SUMMARY = "SUMMARY"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    OCR = "OCR"
    EXTRACTION = "EXTRACTION"
    TABLE_ANALYSIS = "TABLE_ANALYSIS"
    CHART_ANALYSIS = "CHART_ANALYSIS"
    IMAGE_ANALYSIS = "IMAGE_ANALYSIS"
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"
    VISUALIZATION = "VISUALIZATION"
    RETRIEVAL = "RETRIEVAL"
    INSPECT = "INSPECT"


@dataclass
class OCRBoundingBox:
    box: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    text: str
    confidence: float = 1.0


@dataclass
class OCRResult:
    page_number: int = 1
    full_text: str = ""
    boxes: List[OCRBoundingBox] = field(default_factory=list)
    engine_used: str = "rapidocr"
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class TableData:
    page_number: int = 1
    table_index: int = 0
    df: Optional[pd.DataFrame] = None
    raw_matrix: List[List[str]] = field(default_factory=list)
    caption: str = ""
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        if self.df is not None and not self.df.empty:
            return self.df.to_markdown(index=False)
        if self.raw_matrix:
            return "\n".join([" | ".join(row) for row in self.raw_matrix])
        return "(empty table)"


@dataclass
class ProcessedPage:
    page_number: int
    image_path: Optional[Path] = None
    has_selectable_text: bool = False
    native_text: str = ""
    ocr_result: Optional[OCRResult] = None
    tables: List[TableData] = field(default_factory=list)
    visual_description: str = ""
    chart_detected: bool = False
    diagram_detected: bool = False

    @property
    def combined_text(self) -> str:
        """Returns native text if present, otherwise OCR text, or both."""
        parts = []
        if self.native_text.strip():
            parts.append(self.native_text.strip())
        if self.ocr_result and self.ocr_result.full_text.strip():
            # If native text is sparse or absent, include OCR text
            if len(self.native_text.strip()) < 50:
                parts.append(self.ocr_result.full_text.strip())
        return "\n\n".join(parts)


@dataclass
class RetrievalChunk:
    id: str
    text: str
    score: float
    file_name: str = ""
    department: str = ""
    page: Optional[int] = None
    document_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    input_path: Path
    detected_type: str
    task_type: TaskType
    total_pages: int = 1
    pages: List[ProcessedPage] = field(default_factory=list)
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    visual_findings: str = ""
    tables: List[TableData] = field(default_factory=list)
    retrieved_chunks: List[RetrievalChunk] = field(default_factory=list)
    generated_files: List[Path] = field(default_factory=list)
    final_answer: str = ""
    raw_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

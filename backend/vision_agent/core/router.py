"""
Task router for determining processing workflows based on user command, query, and file type.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import TaskType
from vision_agent.utils.file_utils import detect_file_type
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("router")


class TaskRouter:
    @staticmethod
    def route_command(
        command_name: str,
        file_path: Optional[Path] = None,
        question: Optional[str] = None,
    ) -> TaskType:
        """
        Maps CLI command and intent to a specific TaskType.
        """
        cmd = command_name.lower().strip()

        if cmd in ("summarize", "summary"):
            return TaskType.SUMMARY

        elif cmd in ("ask", "qa", "question"):
            return TaskType.QUESTION_ANSWERING

        elif cmd in ("ocr", "read"):
            return TaskType.OCR

        elif cmd in ("extract", "extraction"):
            return TaskType.EXTRACTION

        elif cmd in ("visualize", "plot", "chart"):
            return TaskType.VISUALIZATION

        elif cmd in ("analyze", "analysis"):
            if question:
                return TaskRouter.infer_task_from_query(question)
            if file_path:
                ft = detect_file_type(file_path)
                if ft == "table_data":
                    return TaskType.TABLE_ANALYSIS
                elif ft == "pdf":
                    return TaskType.DOCUMENT_ANALYSIS
                elif ft == "image":
                    return TaskType.IMAGE_ANALYSIS
            return TaskType.DOCUMENT_ANALYSIS

        elif cmd in ("inspect", "status", "info"):
            return TaskType.INSPECT

        # Default fallback
        return TaskType.DOCUMENT_ANALYSIS

    @staticmethod
    def infer_task_from_query(query: str) -> TaskType:
        """Heuristically infers the sub-task from natural language questions."""
        q = query.lower()

        if any(w in q for w in ["chart", "graph", "plot", "histogram", "trend", "peak", "axis"]):
            return TaskType.CHART_ANALYSIS

        if any(w in q for w in ["table", "column", "row", "total", "average", "highest", "lowest", "sum"]):
            return TaskType.TABLE_ANALYSIS

        if any(w in q for w in ["diagram", "cad", "architecture", "flowchart", "schematic"]):
            return TaskType.IMAGE_ANALYSIS

        if any(w in q for w in ["visualize", "plot this", "draw chart", "create graph"]):
            return TaskType.VISUALIZATION

        if any(w in q for w in ["ocr", "raw text", "transcribe", "extract text"]):
            return TaskType.OCR

        return TaskType.QUESTION_ANSWERING

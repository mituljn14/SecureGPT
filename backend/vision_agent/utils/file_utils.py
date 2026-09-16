"""
File system and path manipulation utilities with security checks.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

FileType = Literal["pdf", "image", "table_data", "text", "cad", "unknown"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
PDF_EXTENSIONS = {".pdf"}
DATA_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json"}
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".rtf"}
# Office documents are read as text (slide/paragraph extraction) so the
# pipeline can summarise and ground them like any other document.
OFFICE_EXTENSIONS = {".pptx", ".ppt", ".docx", ".doc"}
CAD_EXTENSIONS = {".dxf", ".dwg", ".step", ".stp", ".iges", ".igs"}


def validate_file_path(file_path: str | Path) -> Path:
    """
    Validates that a file path exists, is a file, and is not empty.
    Raises ValueError or FileNotFoundError if invalid.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty (0 bytes): {file_path}")
    return path


def detect_file_type(file_path: str | Path) -> FileType:
    """Detect the category of file based on extension and magic signatures."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in DATA_EXTENSIONS:
        return "table_data"
    if ext in TEXT_EXTENSIONS or ext in OFFICE_EXTENSIONS:
        return "text"
    if ext in CAD_EXTENSIONS:
        return "cad"

    return "unknown"


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe cross-platform filenames."""
    name = re.sub(r'[\\/*?:"<>| ]', "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def safe_output_path(target_dir: Path, base_name: str, extension: str) -> Path:
    """
    Generates a collision-safe output path inside target_dir.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(base_name)
    clean_ext = extension if extension.startswith(".") else f".{extension}"

    candidate = target_dir / f"{clean_name}{clean_ext}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = target_dir / f"{clean_name}_{counter}{clean_ext}"
        if not candidate.exists():
            return candidate
        counter += 1

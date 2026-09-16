"""
Logging and Tee-output utilities for the Vision Agent.
Allows output to be simultaneously printed to terminal and appended to a session log file.
"""
from __future__ import annotations

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO

from vision_agent.config.settings import settings


class TeeWriter:
    """
    Duplicates writes to both standard output and a file.
    Provides real-time 'tee' functionality.
    """

    def __init__(self, stream: TextIO, file_path: Path):
        self.stream = stream
        self.file_path = file_path
        self.file: TextIO | None = None
        try:
            self.file = open(file_path, "a", encoding="utf-8", buffering=1)
        except Exception as exc:
            sys.stderr.write(f"Warning: Could not open log file {file_path}: {exc}\n")

    def write(self, data: str) -> int:
        written = self.stream.write(data)
        self.stream.flush()
        if self.file:
            try:
                self.file.write(data)
                self.file.flush()
            except Exception:
                pass
        return written

    def flush(self) -> None:
        self.stream.flush()
        if self.file:
            try:
                self.file.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()


_tee_installed = False


def setup_tee_logger(log_prefix: str = "vision_agent") -> Path:
    """
    Installs global stdout/stderr teeing to a timestamped log file.
    Returns the log file path.
    """
    global _tee_installed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = settings.LOGS_DIR / f"{log_prefix}_{timestamp}.log"
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not _tee_installed:
        sys.stdout = TeeWriter(sys.stdout, log_file)
        sys.stderr = TeeWriter(sys.stderr, log_file)
        _tee_installed = True

    return log_file


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a standard Python logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_name = settings.LOG_LEVEL.upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False
    return logger

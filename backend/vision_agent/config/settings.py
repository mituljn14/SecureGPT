"""
Configuration settings for the Production Vision Agent.
Loads configuration from environment variables and .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Locate project root and load .env if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Project Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    INPUTS_DIR: Path = PROJECT_ROOT / "inputs"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    SUMMARIES_DIR: Path = PROJECT_ROOT / "outputs" / "summaries"
    EXTRACTED_DIR: Path = PROJECT_ROOT / "outputs" / "extracted"
    VISUALIZATIONS_DIR: Path = PROJECT_ROOT / "outputs" / "visualizations"
    TEMP_DIR: Path = PROJECT_ROOT / "outputs" / "temporary"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Ollama Vision & Text LLM Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "moondream")
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "llama3.2")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Qdrant Vector Database Settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "securegpt_v2")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
    QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "30"))

    # RBAC identity used for corpus retrieval when the caller does not pass
    # one explicitly (standalone CLI / desktop GUI). Deliberately EMPTY by
    # default: with no role we skip corpus retrieval entirely rather than
    # search everything. Set VISION_AGENT_ROLE (e.g. "COO") to ground
    # standalone runs against the corpus.
    DEFAULT_ROLE: str = os.getenv("VISION_AGENT_ROLE", "")
    DEFAULT_DEPARTMENT: str | None = os.getenv("VISION_AGENT_DEPARTMENT", None)

    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

    # Vision & Image Processing Constraints
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "1280"))
    IMAGE_JPEG_QUALITY: int = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))

    # OCR Settings
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "rapidocr")  # 'rapidocr' or 'tesseract'

    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    TEE_LOGGING: bool = True

    def ensure_directories(self) -> None:
        """Ensure all runtime directories exist."""
        for directory in [
            self.INPUTS_DIR,
            self.OUTPUTS_DIR,
            self.SUMMARIES_DIR,
            self.EXTRACTED_DIR,
            self.VISUALIZATIONS_DIR,
            self.TEMP_DIR,
            self.LOGS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

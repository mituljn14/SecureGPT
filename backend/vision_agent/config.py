"""
Configuration for the Local Vision Agent.

Step 1 scope: only the settings needed for environment verification
and read-only Qdrant/Ollama inspection. More settings are added in
later steps as each component is built.
"""
import os
from pathlib import Path

# --- Ollama ---
OLLAMA_HOST = os.environ.get("VISION_AGENT_OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL_NAME = os.environ.get("VISION_AGENT_MODEL", "qwen2.5vl:3b")

# --- Qdrant ---
QDRANT_HOST = os.environ.get("VISION_AGENT_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("VISION_AGENT_QDRANT_PORT", "6333"))
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Leave this as None until Step 1 output tells you which collection to use.
# If you already know the collection name, you can set it via the
# VISION_AGENT_QDRANT_COLLECTION environment variable instead of editing this file.
QDRANT_COLLECTION_NAME = os.environ.get("VISION_AGENT_QDRANT_COLLECTION", None)

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# --- Inspection safety limits ---
# Keep payload previews small so we never dump full document contents to the terminal.
PAYLOAD_SAMPLE_LIMIT = 3
PAYLOAD_TEXT_PREVIEW_CHARS = 200

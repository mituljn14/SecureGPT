"""
Comprehensive environment and dependency verification test.
"""
from __future__ import annotations

import sys
import platform
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.config.settings import settings
from vision_agent.vision.ollama_client import OllamaClient
from vision_agent.retrieval.qdrant_client import get_qdrant_wrapper


def test_python_version():
    ver = platform.python_version()
    assert sys.version_info >= (3, 9), f"Python version too old: {ver}"
    print(f"[OK] Python Version: {ver}")


def test_ollama_connectivity():
    client = OllamaClient()
    connected = client.check_connection()
    print(f"[OK] Ollama Server at {client.base_url}: {'CONNECTED' if connected else 'DISCONNECTED'}")
    assert connected, f"Ollama is unreachable at {client.base_url}"


def test_vision_model_availability():
    client = OllamaClient()
    available, msg = client.verify_model_available()
    print(f"[OK] Vision Model ({settings.VISION_MODEL}): {'AVAILABLE' if available else 'MISSING'}")
    assert available, msg


def test_qdrant_connectivity():
    wrapper = get_qdrant_wrapper()
    connected = wrapper.check_connection()
    print(f"[OK] Qdrant Server at {settings.QDRANT_URL}: {'CONNECTED' if connected else 'DISCONNECTED'}")
    assert connected, f"Qdrant is unreachable at {settings.QDRANT_URL}"


def test_qdrant_collections():
    wrapper = get_qdrant_wrapper()
    collections = wrapper.list_collections()
    print(f"[OK] Qdrant Collections found: {len(collections)}")
    for c in collections:
        print(f"   - {c['name']} (points: {c['points_count']}, dim: {c['dimension']})")
    assert len(collections) > 0, "No collections found in Qdrant."


if __name__ == "__main__":
    print("=" * 40)
    print("RUNNING ENVIRONMENT VERIFICATION")
    print("=" * 40)
    test_python_version()
    test_ollama_connectivity()
    test_vision_model_availability()
    test_qdrant_connectivity()
    test_qdrant_collections()
    print("\nALL ENVIRONMENT CHECKS PASSED.")

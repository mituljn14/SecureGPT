"""
Test Ollama API and Vision Model inference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.vision.ollama_client import OllamaClient
from vision_agent.utils.image_utils import encode_image_to_base64


def test_ollama_text_generation():
    client = OllamaClient()
    assert client.check_connection(), "Ollama is not running"
    resp = client.generate("Respond with the single word: READY")
    assert resp and len(resp) > 0, "Empty Ollama response"
    print(f"[OK] Ollama Text Generation Output: {resp}")


def test_ollama_vision_inference():
    img = Image.new("RGB", (150, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 130, 130], fill=(255, 0, 0))

    b64_img = encode_image_to_base64(img)
    client = OllamaClient()
    resp = client.generate(
        prompt="Describe this image.",
        images_base64=[b64_img],
    )
    assert resp and len(resp) > 0, "Empty vision response"
    print(f"[OK] Ollama Vision Inference Output: {resp.strip()[:100]}...")


if __name__ == "__main__":
    test_ollama_text_generation()
    test_ollama_vision_inference()
    print("Ollama tests passed successfully.")

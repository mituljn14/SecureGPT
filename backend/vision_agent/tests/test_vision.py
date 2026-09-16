"""
Test high-level Vision Agent capabilities.
"""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.vision.vision_agent import VisionAgent


def test_vision_chart_analysis():
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 80, 100, 180], fill=(255, 0, 0))  # Bar 1
    draw.rectangle([150, 40, 200, 180], fill=(0, 255, 0))  # Bar 2
    draw.text((60, 185), "Q1", fill=(0, 0, 0))
    draw.text((160, 185), "Q2", fill=(0, 0, 0))

    agent = VisionAgent()
    analysis = agent.analyze_chart(img)
    print(f"[OK] Chart Analysis Output:\n{analysis}")
    assert analysis and len(analysis) > 5


def test_vision_diagram_explanation():
    img = Image.new("RGB", (300, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 40, 100, 100], outline=(0, 0, 0), width=2)
    draw.text((40, 60), "Input", fill=(0, 0, 0))
    draw.line([(100, 70), (180, 70)], fill=(0, 0, 0), width=2)
    draw.rectangle([180, 40, 260, 100], outline=(0, 0, 0), width=2)
    draw.text((190, 60), "Output", fill=(0, 0, 0))

    agent = VisionAgent()
    explanation = agent.explain_diagram(img)
    print(f"[OK] Diagram Explanation Output:\n{explanation}")
    assert explanation and len(explanation) > 5


if __name__ == "__main__":
    test_vision_chart_analysis()
    test_vision_diagram_explanation()
    print("Vision tests passed successfully.")

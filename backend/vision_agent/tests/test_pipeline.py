"""
End-to-End Pipeline test verifying summarizing, QA, tabular analysis, and visualization.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.core.pipeline import VisionPipeline
from vision_agent.config.settings import settings
from tests.test_pdf import create_synthetic_pdf


def test_pipeline_on_pdf():
    test_pdf = settings.TEMP_DIR / "pipeline_test.pdf"
    create_synthetic_pdf(test_pdf)

    pipeline = VisionPipeline()
    res = pipeline.run(test_pdf, command="summarize")

    print(f"[OK] Pipeline Run on PDF: Total Pages = {res.total_pages}")
    print(f"[OK] Summary generated: {bool(res.final_answer)}")
    assert res.total_pages == 2
    assert len(res.generated_files) > 0


def test_pipeline_on_csv():
    test_csv = settings.TEMP_DIR / "sample_sales.csv"
    df = pd.DataFrame(
        {
            "Region": ["North", "South", "East", "West"],
            "Revenue": [45000, 52000, 31000, 68000],
            "Growth": [12.5, 8.2, -3.1, 19.4],
        }
    )
    df.to_csv(test_csv, index=False)

    pipeline = VisionPipeline()
    res = pipeline.run(test_csv, command="visualize")

    print(f"[OK] Pipeline Run on CSV: Generated files = {len(res.generated_files)}")
    assert len(res.generated_files) > 0
    assert any(str(f).endswith(".png") for f in res.generated_files), "Chart PNG was not generated"


if __name__ == "__main__":
    test_pipeline_on_pdf()
    test_pipeline_on_csv()
    print("Pipeline end-to-end tests passed successfully.")

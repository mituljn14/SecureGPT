"""
High-level Vision Agent wrapping Ollama and image processing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional
from PIL import Image

from .ollama_client import OllamaClient
from .image_processor import ImageProcessor
from .prompts import VisionPrompts
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("vision_agent")


class VisionAgent:
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        image_processor: Optional[ImageProcessor] = None,
    ):
        self.ollama = ollama_client or OllamaClient()
        self.img_proc = image_processor or ImageProcessor()

    def describe_image(self, source: Union[str, Path, bytes, Image.Image]) -> str:
        """Generate a comprehensive factual description of the image."""
        _, b64 = self.img_proc.prepare_image(source)
        return self.ollama.generate(
            prompt=VisionPrompts.IMAGE_SUMMARY,
            images_base64=[b64],
        )

    def analyze_chart(self, source: Union[str, Path, bytes, Image.Image]) -> str:
        """Analyze a chart, graph, or plot visual."""
        _, b64 = self.img_proc.prepare_image(source)
        return self.ollama.generate(
            prompt=VisionPrompts.CHART_ANALYSIS,
            images_base64=[b64],
        )

    def explain_diagram(self, source: Union[str, Path, bytes, Image.Image]) -> str:
        """Explain a technical diagram, flowchart, or CAD rendering."""
        _, b64 = self.img_proc.prepare_image(source)
        return self.ollama.generate(
            prompt=VisionPrompts.DIAGRAM_ANALYSIS,
            images_base64=[b64],
        )

    def understand_table(self, source: Union[str, Path, bytes, Image.Image]) -> str:
        """Read and structure an image of a table."""
        _, b64 = self.img_proc.prepare_image(source)
        return self.ollama.generate(
            prompt=VisionPrompts.TABLE_UNDERSTANDING,
            images_base64=[b64],
        )

    def answer_question(
        self,
        source: Union[str, Path, bytes, Image.Image],
        question: str,
        context: str = "",
    ) -> str:
        """Answer a question about the image with optional retrieved context."""
        _, b64 = self.img_proc.prepare_image(source)
        prompt = VisionPrompts.visual_qa(question=question, context=context)
        return self.ollama.generate(
            prompt=prompt,
            images_base64=[b64],
        )

    def synthesize_grounded_answer(
        self,
        question: str,
        document_text: str,
        visual_findings: str,
        retrieved_chunks: str,
    ) -> str:
        """Synthesize final grounded response using LLM reasoning."""
        prompt = VisionPrompts.grounded_synthesis(
            question=question,
            document_text=document_text,
            visual_findings=visual_findings,
            retrieved_chunks=retrieved_chunks,
        )
        return self.ollama.generate(prompt=prompt)

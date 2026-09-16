"""
Curated, high-precision prompts for Vision LLM tasks.
"""
from __future__ import annotations


class VisionPrompts:
    IMAGE_SUMMARY = (
        "Describe this image thoroughly. Identify the main subject, text, objects, layout, "
        "and any significant details. Be factual and clear."
    )

    CHART_ANALYSIS = (
        "Analyze this chart or graph in detail:\n"
        "1. Identify the chart type (e.g., Bar, Line, Pie, Scatter, Dashboard).\n"
        "2. State the chart title, legend, and axis labels (X-axis, Y-axis with units).\n"
        "3. Extract key data points, values, highs, lows, and trends.\n"
        "4. Summarize the main analytical takeaway."
    )

    DIAGRAM_ANALYSIS = (
        "Analyze this diagram, technical drawing, or CAD visualization:\n"
        "1. Identify the purpose of the diagram / system depicted.\n"
        "2. Trace key components, blocks, flows, inputs, and outputs.\n"
        "3. Read any annotations, labels, dimensions, or callouts.\n"
        "4. Provide a clear, step-by-step explanation of what is shown."
    )

    TABLE_UNDERSTANDING = (
        "Inspect this table. Extract the column headers, row labels, and numeric data values. "
        "Format the output as a clean markdown table if possible, followed by key observations."
    )

    DOCUMENT_PAGE_ANALYSIS = (
        "Analyze this document page. Describe the visual layout, any figures, logos, headers, "
        "stamps, handwritten notes, charts, or tables present, and how they relate to the page content."
    )

    @staticmethod
    def visual_qa(question: str, context: str = "") -> str:
        prompt = f"Answer the following question about this image/document accurately and concisely:\nQuestion: {question}"
        if context:
            prompt += f"\n\nAdditional Text / Retrieved Knowledge:\n{context}"
        return prompt

    @staticmethod
    def grounded_synthesis(question: str, document_text: str, visual_findings: str, retrieved_chunks: str) -> str:
        return (
            f"You are a helpful and precise Vision & Document Agent. Answer the user request grounded strictly "
            f"on the provided document content, visual observations, and retrieved organizational knowledge.\n\n"
            f"--- DOCUMENT TEXT ---\n{document_text}\n\n"
            f"--- VISUAL FINDINGS ---\n{visual_findings}\n\n"
            f"--- RETRIEVED KNOWLEDGE ---\n{retrieved_chunks}\n\n"
            f"--- USER REQUEST ---\n{question}\n\n"
            f"Provide a clear, structured, and factual response. Cite page numbers or source files when referencing facts."
        )

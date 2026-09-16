"""
Central Vision Pipeline orchestrating input detection, preprocessing, OCR,
vision inference, Qdrant retrieval, tabular analytics, visualization, and synthesis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from vision_agent.config.settings import settings
from vision_agent.core.models import TaskType, PipelineResult, ProcessedPage, TableData, RetrievalChunk
from vision_agent.core.router import TaskRouter
from vision_agent.document.document_analyzer import DocumentAnalyzer
from vision_agent.vision.vision_agent import VisionAgent
from vision_agent.retrieval.retriever import KnowledgeRetriever
from vision_agent.data.data_analyzer import DataAnalyzer
from vision_agent.visualization.chart_generator import ChartGenerator
from vision_agent.utils.file_utils import validate_file_path, safe_output_path
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("pipeline")


class VisionPipeline:
    def __init__(
        self,
        document_analyzer: Optional[DocumentAnalyzer] = None,
        vision_agent: Optional[VisionAgent] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        chart_generator: Optional[ChartGenerator] = None,
        role: Optional[str] = None,
        department: Optional[str] = None,
    ):
        self.doc_analyzer = document_analyzer or DocumentAnalyzer()
        self.vision = vision_agent or VisionAgent()
        self.retriever = retriever or KnowledgeRetriever()
        self.chart_gen = chart_generator or ChartGenerator()
        # Falls back to settings so the standalone CLI/GUI still work.
        self.role = role or settings.DEFAULT_ROLE or None
        self.department = department or settings.DEFAULT_DEPARTMENT

    def run(
        self,
        file_path: str | Path,
        command: str = "analyze",
        question: Optional[str] = None,
        force_ocr: bool = False,
        role: Optional[str] = None,
        department: Optional[str] = None,
    ) -> PipelineResult:
        """
        Executes end-to-end processing pipeline for a given document/image and task.
        """
        path = validate_file_path(file_path)
        task_type = TaskRouter.route_command(command, file_path=path, question=question)
        logger.info(f"Initiating pipeline for '{path.name}' [Task: {task_type.value}]")

        generated_files: List[Path] = []
        key_findings: List[str] = []
        visual_findings_list: List[str] = []
        retrieved_chunks: List[RetrievalChunk] = []

        # 1. Document & Page Ingestion (Parsing, OCR, Tables)
        detected_type, pages, tables = self.doc_analyzer.analyze(path, force_ocr=force_ocr)
        logger.info(f"Ingested {len(pages)} page(s) / elements. Tables found: {len(tables)}.")

        # 2. Vision Understanding (Page / Image visual inference)
        for page in pages:
            if page.image_path:
                try:
                    if task_type in (TaskType.CHART_ANALYSIS, TaskType.VISUALIZATION):
                        vis_desc = self.vision.analyze_chart(page.image_path)
                    elif task_type == TaskType.IMAGE_ANALYSIS:
                        vis_desc = self.vision.explain_diagram(page.image_path)
                    elif task_type == TaskType.QUESTION_ANSWERING and question:
                        vis_desc = self.vision.answer_question(page.image_path, question)
                    else:
                        vis_desc = self.vision.describe_image(page.image_path)

                    page.visual_description = vis_desc
                    if vis_desc.strip():
                        visual_findings_list.append(f"Page {page.page_number}: {vis_desc}")
                except Exception as exc:
                    logger.warning(f"Vision inference failed for Page {page.page_number}: {exc}")
                    page.visual_description = f"(Vision analysis unavailable: {exc})"

        # 3. Tabular & Numerical Analysis (Pandas & NumPy)
        all_dfs: List[pd.DataFrame] = [tbl.df for tbl in tables if tbl.df is not None and not tbl.df.empty]
        combined_df: Optional[pd.DataFrame] = None

        if all_dfs:
            combined_df = all_dfs[0]
            for tbl in tables:
                if tbl.df is not None and not tbl.df.empty:
                    analysis_res = DataAnalyzer.analyze_table(tbl.df)
                    tbl.summary_stats = analysis_res.get("summary", {})
                    key_findings.extend(analysis_res.get("insights", []))

                    # Save extracted table CSV
                    tbl_csv_path = safe_output_path(
                        settings.EXTRACTED_DIR,
                        f"{path.stem}_table_p{tbl.page_number}_{tbl.table_index}",
                        ".csv",
                    )
                    tbl.df.to_csv(tbl_csv_path, index=False)
                    generated_files.append(tbl_csv_path)

        # 4. Automatic Visualization (if requested or useful)
        if task_type in (TaskType.VISUALIZATION, TaskType.CHART_ANALYSIS) or (command == "visualize" and combined_df is not None):
            if combined_df is not None and not combined_df.empty:
                chart_path = self.chart_gen.auto_visualize(combined_df, title_hint=f"{path.stem} Data Overview")
                if chart_path:
                    generated_files.append(chart_path)
                    key_findings.append(f"Generated visualization chart saved to: {chart_path.name}")

        # 5. Qdrant Knowledge Retrieval (if question or summarization context needed)
        query_for_retrieval = question or (f"{path.stem} " + " ".join([p.combined_text[:100] for p in pages[:2]]))
        effective_role = role or self.role
        effective_dept = department if department is not None else self.department
        if query_for_retrieval.strip():
            if effective_role:
                retrieved_chunks = self.retriever.retrieve(
                    query_for_retrieval, role=effective_role,
                    department=effective_dept, top_k=3,
                )
            else:
                # No role -> skip corpus grounding rather than search across
                # every department. Document analysis, OCR, tables, vision
                # and charts all still run; only the Qdrant citations drop.
                logger.warning(
                    "No role supplied (set VISION_AGENT_ROLE or pass role=) -- "
                    "skipping corpus retrieval for this run."
                )

        # 6. Response Synthesis & Grounded Summary
        combined_doc_text = "\n\n".join(
            [f"--- Page {p.page_number} ---\n{p.combined_text}" for p in pages if p.combined_text.strip()]
        )
        combined_visuals = "\n\n".join(visual_findings_list)
        retrieved_text_formatted = "\n\n".join(
            [
                f"[Source: {c.file_name} | Page: {c.page or 'N/A'} | Score: {c.score:.3f}]\n{c.text}"
                for c in retrieved_chunks
            ]
        )

        final_answer = ""
        summary_text = ""

        if task_type == TaskType.QUESTION_ANSWERING and question:
            try:
                final_answer = self.vision.synthesize_grounded_answer(
                    question=question,
                    document_text=combined_doc_text[:4000],
                    visual_findings=combined_visuals[:2000],
                    retrieved_chunks=retrieved_text_formatted[:2000],
                )
            except Exception as exc:
                final_answer = (
                    f"Answer based on extracted content:\n"
                    f"{combined_doc_text[:1000]}\n\n"
                    f"Visual observations: {combined_visuals[:500]}"
                )

        elif task_type in (TaskType.SUMMARY, TaskType.DOCUMENT_ANALYSIS):
            prompt = (
                f"Summarize the key information, data, and findings from this document succinctly and factually:\n\n"
                f"{combined_doc_text[:4000]}\n\n"
                f"Visual Observations:\n{combined_visuals[:2000]}"
            )
            try:
                summary_text = self.vision.ollama.generate(prompt=prompt)
            except Exception as exc:
                summary_text = combined_doc_text[:1000] if combined_doc_text else combined_visuals[:1000]

            final_answer = summary_text

        elif task_type == TaskType.OCR:
            ocr_texts = []
            for p in pages:
                if p.ocr_result and p.ocr_result.full_text:
                    ocr_texts.append(f"--- Page {p.page_number} OCR Text ---\n{p.ocr_result.full_text}")
                elif p.native_text:
                    ocr_texts.append(f"--- Page {p.page_number} Native Text ---\n{p.native_text}")
            final_answer = "\n\n".join(ocr_texts) if ocr_texts else "No text detected in document."

        elif task_type == TaskType.EXTRACTION:
            extraction_parts = []
            if tables:
                for tbl in tables:
                    extraction_parts.append(f"### {tbl.caption}\n{tbl.to_markdown()}")
            if combined_doc_text:
                extraction_parts.append(f"### Extracted Text Content\n{combined_doc_text[:2000]}")
            final_answer = "\n\n".join(extraction_parts)

        else:
            final_answer = combined_visuals if combined_visuals else combined_doc_text[:1000]

        # Save summary report artifact
        if final_answer:
            summary_file = safe_output_path(settings.SUMMARIES_DIR, f"{path.stem}_summary", ".md")
            summary_file.write_text(
                f"# Analysis Summary: {path.name}\n\n"
                f"**Task Type:** {task_type.value}\n"
                f"**Pages Processed:** {len(pages)}\n\n"
                f"## Final Answer / Summary\n{final_answer}\n\n"
                f"## Visual Findings\n{combined_visuals}\n\n"
                f"## Key Insights\n" + "\n".join([f"- {k}" for k in key_findings]),
                encoding="utf-8",
            )
            generated_files.append(summary_file)

        return PipelineResult(
            input_path=path,
            detected_type=detected_type.upper(),
            task_type=task_type,
            total_pages=len(pages),
            pages=pages,
            summary=summary_text,
            key_findings=key_findings,
            visual_findings=combined_visuals,
            tables=tables,
            retrieved_chunks=retrieved_chunks,
            generated_files=generated_files,
            final_answer=final_answer,
            raw_data=combined_df,
        )

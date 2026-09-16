"""
Command line interface dispatcher and result formatter for Vision Agent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from vision_agent.config.settings import settings
from vision_agent.core.pipeline import VisionPipeline
from vision_agent.core.models import PipelineResult
from vision_agent.retrieval.qdrant_client import get_qdrant_wrapper
from vision_agent.vision.ollama_client import OllamaClient
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("cli")


class CLIHandler:
    def __init__(self):
        self.pipeline = VisionPipeline()

    @staticmethod
    def print_header(title: str = "VISION AGENT RESULT"):
        print("=" * 60)
        print(f" {title.center(56)} ")
        print("=" * 60)

    @staticmethod
    def format_output(result: PipelineResult):
        """Formats and displays the structured result according to specification."""
        print()
        CLIHandler.print_header("VISION AGENT RESULT")
        print()
        print(f"Input:\n  {result.input_path.name}")
        print()
        print(f"Detected Type:\n  {result.detected_type}")
        print()

        # Compute processing steps applied
        proc_steps = ["File Ingestion"]
        has_native = any(p.has_selectable_text for p in result.pages)
        has_ocr = any(p.ocr_result is not None and p.ocr_result.success for p in result.pages)
        has_vision = any(bool(p.visual_description) for p in result.pages)
        has_retrieval = len(result.retrieved_chunks) > 0
        has_data = len(result.tables) > 0 or result.raw_data is not None

        if has_native:
            proc_steps.append("Text Extraction")
        if has_ocr:
            proc_steps.append("OCR")
        if has_vision:
            proc_steps.append("Vision LLM")
        if has_data:
            proc_steps.append("Pandas Analytics")
        if has_retrieval:
            proc_steps.append("Qdrant Retrieval")

        print(f"Processing:\n  {' + '.join(proc_steps)}")
        print()
        print(f"Pages:\n  {result.total_pages}")
        print()

        if result.final_answer:
            header_name = "Summary:" if result.task_type.value == "SUMMARY" else "Answer / Summary:"
            print(f"{header_name}\n  {result.final_answer}")
            print()

        if result.key_findings:
            print("Key Findings:")
            for i, finding in enumerate(result.key_findings, 1):
                print(f"  {i}. {finding}")
            print()

        if result.retrieved_chunks:
            print("Retrieved Context (from Qdrant):")
            for c in result.retrieved_chunks:
                source_info = f"Document: {c.file_name} | Page: {c.page or 'N/A'} | Dept: {c.department or 'N/A'} (Score: {c.score:.3f})"
                preview = c.text.strip().replace("\n", " ")[:150]
                print(f"  - [{source_info}]\n    \"{preview}...\"")
            print()

        if result.visual_findings:
            print("Visual Findings:")
            for line in result.visual_findings.splitlines():
                if line.strip():
                    print(f"  {line.strip()}")
            print()

        if result.generated_files:
            print("Generated Files:")
            for gf in result.generated_files:
                try:
                    rel_path = gf.relative_to(settings.PROJECT_ROOT)
                    print(f"  - {rel_path}")
                except Exception:
                    print(f"  - {gf}")
            print()

        print("=" * 60)
        print()

    def handle_summarize(self, file_path: str):
        result = self.pipeline.run(file_path, command="summarize")
        self.format_output(result)

    def handle_analyze(self, file_path: str):
        result = self.pipeline.run(file_path, command="analyze")
        self.format_output(result)

    def handle_ask(self, file_path: str, question: str):
        result = self.pipeline.run(file_path, command="ask", question=question)
        self.format_output(result)

    def handle_direct_query(self, query: str, top_k: int = 4):
        """
        Answers a user query directly by searching across all documents in Qdrant.
        """
        print()
        CLIHandler.print_header("KNOWLEDGE BASE QUERY RESULT")
        print()
        print(f"User Query:\n  {query}\n")

        # 1. Retrieve relevant chunks from Qdrant
        # role comes from the pipeline (VISION_AGENT_ROLE or an explicit
        # --role); without one we cannot query the corpus, because
        # retrieval is RBAC-filtered by design.
        if not self.pipeline.role:
            print("No role set. Export VISION_AGENT_ROLE (e.g. COO) or pass "
                  "--role to query the enterprise corpus.")
            return
        retrieved_chunks = self.pipeline.retriever.retrieve(
            query, role=self.pipeline.role,
            department=self.pipeline.department, top_k=top_k)

        if not retrieved_chunks:
            print("Answer:\n  No relevant information found in the Qdrant knowledge base for this query.\n")
            print("=" * 60)
            print()
            return

        formatted_context = "\n\n".join(
            [
                f"[Source: {c.file_name} | Page: {c.page or 'N/A'} | Dept: {c.department or 'N/A'} | Score: {c.score:.3f}]\n{c.text}"
                for c in retrieved_chunks
            ]
        )

        # 2. Synthesize grounded answer
        prompt = (
            f"You are the SecureGPT Knowledge Agent. Answer the user query based strictly on the retrieved knowledge below.\n\n"
            f"--- RETRIEVED KNOWLEDGE ---\n{formatted_context}\n\n"
            f"--- USER QUERY ---\n{query}\n\n"
            f"Provide a clear, accurate, and concise response. Cite the source document or department when relevant."
        )

        try:
            answer = self.pipeline.vision.ollama.generate(prompt=prompt)
        except Exception as exc:
            answer = f"Generated directly from retrieved chunks:\n\n{retrieved_chunks[0].text[:800]}"

        print(f"Answer:\n  {answer}\n")

        print("Retrieved Sources (from Qdrant securegpt_corpus):")
        for c in retrieved_chunks:
            source_info = f"Document: {c.file_name} | Page: {c.page or 'N/A'} | Dept: {c.department or 'N/A'} (Score: {c.score:.3f})"
            snippet = c.text.strip().replace("\n", " ")[:160]
            print(f"  - [{source_info}]\n    \"{snippet}...\"")
        print()

        # Save query result artifact
        from vision_agent.utils.file_utils import safe_output_path
        out_file = safe_output_path(settings.SUMMARIES_DIR, "query_answer", ".md")
        out_file.write_text(
            f"# Query Result\n\n**Query:** {query}\n\n## Answer\n{answer}\n\n## Sources\n"
            + "\n".join([f"- **{c.file_name}** (Page: {c.page or 'N/A'}, Dept: {c.department}, Score: {c.score:.3f})" for c in retrieved_chunks]),
            encoding="utf-8",
        )
        try:
            rel = out_file.relative_to(settings.PROJECT_ROOT)
            print(f"Saved Answer to: {rel}\n")
        except Exception:
            print(f"Saved Answer to: {out_file}\n")

        print("=" * 60)
        print()

    def handle_extract(self, file_path: str):
        result = self.pipeline.run(file_path, command="extract")
        self.format_output(result)

    def handle_visualize(self, file_path: str):
        result = self.pipeline.run(file_path, command="visualize")
        self.format_output(result)

    def handle_inspect(self, file_path: Optional[str] = None):
        """Inspects environment, Ollama, and Qdrant connectivity."""
        print()
        CLIHandler.print_header("SYSTEM INSPECTION & ENVIRONMENT")
        print()

        # 1. Ollama status
        ollama = OllamaClient()
        ollama_ok = ollama.check_connection()
        print(f"Ollama Server ({settings.OLLAMA_BASE_URL}): {'AVAILABLE' if ollama_ok else 'UNAVAILABLE'}")
        if ollama_ok:
            models = ollama.list_models()
            print(f"  Installed Models: {', '.join(models) if models else 'None'}")
            model_avail, model_msg = ollama.verify_model_available()
            print(f"  Active Vision Model ({settings.VISION_MODEL}): {'READY' if model_avail else 'MISSING'}")
            if not model_avail:
                print(f"  {model_msg}")
        else:
            print("  Warning: Start Ollama with `ollama serve`")
        print()

        # 2. Qdrant status
        qdrant = get_qdrant_wrapper()
        qdrant_ok = qdrant.check_connection()
        print(f"Qdrant Database ({settings.QDRANT_URL}): {'AVAILABLE' if qdrant_ok else 'UNAVAILABLE'}")
        if qdrant_ok:
            try:
                collections = qdrant.list_collections()
                print(f"  Found {len(collections)} collection(s):")
                for c in collections:
                    is_active = "(active in .env)" if c["name"] == settings.QDRANT_COLLECTION else ""
                    print(
                        f"    - {c['name']} {is_active}: {c['points_count']} points, "
                        f"dim: {c['dimension']}, distance: {c['distance']}"
                    )
            except Exception as exc:
                print(f"  Error inspecting collections: {exc}")
        else:
            print("  Warning: Qdrant is not reachable at configured URL.")
        print()

        # 3. If file given, inspect file
        if file_path:
            p = Path(file_path)
            print(f"File Inspection: {p.name}")
            if p.exists():
                print(f"  Size: {p.stat().st_size:,} bytes")
                from vision_agent.utils.file_utils import detect_file_type

                print(f"  Detected Type: {detect_file_type(p)}")
            else:
                print("  File does not exist.")
            print()

        print("=" * 60)
        print()

"""
Lightweight native Desktop UI for the Local Vision Agent.
Built with standard library Tkinter + Pillow (no web servers, no open ports).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from vision_agent.config.settings import settings
from vision_agent.core.pipeline import VisionPipeline
from vision_agent.core.models import PipelineResult
from vision_agent.cli.commands import CLIHandler
from vision_agent.utils.file_utils import detect_file_type


class VisionAgentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Local Vision & Document Intelligence Agent")
        self.root.geometry("1020x720")
        self.root.minsize(850, 600)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.pipeline = VisionPipeline()
        self.cli_handler = CLIHandler()

        self.current_preview_image: Optional[ImageTk.PhotoImage] = None

        self._setup_ui()

    def _setup_ui(self):
        # Top Header Banner
        header = tk.Frame(self.root, bg="#1e293b", height=65)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="Vision Agent — Multimodal & Document Intelligence",
            font=("Helvetica", 14, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        )
        title_lbl.pack(side=tk.LEFT, padx=18, pady=12)

        model_info = tk.Label(
            header,
            text=f"Vision: {settings.VISION_MODEL} | Vector DB: Qdrant",
            font=("Helvetica", 9),
            fg="#94a3b8",
            bg="#1e293b",
        )
        model_info.pack(side=tk.RIGHT, padx=18, pady=16)

        # Main Content Frame
        main_frame = tk.Frame(self.root, padx=15, pady=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control Panel (Top Controls)
        ctrl_frame = ttk.LabelFrame(main_frame, text=" Task Controls & Input ", padding=10)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Operation Selection
        lbl_op = ttk.Label(ctrl_frame, text="Select Operation:", font=("Helvetica", 9, "bold"))
        lbl_op.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.op_var = tk.StringVar(value="Summarize Document / Image")
        self.op_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.op_var,
            state="readonly",
            values=[
                "Summarize Document / Image",
                "Ask Question (Document VQA + Qdrant)",
                "Direct Query (Search Knowledge Base)",
                "Data Visualization & Chart Generation",
                "CAD Blueprint & Diagram Analysis",
                "Extract Tables to CSV",
                "Deep Multimodal Analysis",
                "System & Qdrant Inspection",
            ],
            width=38,
        )
        self.op_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.op_combo.bind("<<ComboboxSelected>>", self._on_operation_change)

        # Row 1: File Selection Frame
        lbl_file = ttk.Label(ctrl_frame, text="Target File:", font=("Helvetica", 9, "bold"))
        lbl_file.grid(row=0, column=2, sticky=tk.W, padx=(15, 5), pady=5)

        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(ctrl_frame, textvariable=self.file_var, width=32)
        self.file_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        self.btn_browse = ttk.Button(ctrl_frame, text="Browse...", command=self._browse_file)
        self.btn_browse.grid(row=0, column=4, padx=5, pady=5)

        # Row 2: Question / Query Field
        self.lbl_query = ttk.Label(ctrl_frame, text="Question / Query:", font=("Helvetica", 9, "bold"))
        self.lbl_query.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(ctrl_frame, textvariable=self.query_var, width=70)
        self.query_entry.grid(row=1, column=1, columnspan=3, sticky=tk.W + tk.E, padx=5, pady=5)

        # Action Buttons
        btn_frame = tk.Frame(ctrl_frame)
        btn_frame.grid(row=1, column=4, padx=5, pady=5, sticky=tk.E)

        self.btn_run = tk.Button(
            btn_frame,
            text="Run Operation",
            command=self._start_execution,
            bg="#0284c7",
            fg="white",
            font=("Helvetica", 9, "bold"),
            padx=12,
            pady=4,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self.btn_run.pack(side=tk.LEFT)

        # Status Line
        self.status_var = tk.StringVar(value="Ready.")
        status_lbl = tk.Label(main_frame, textvariable=self.status_var, anchor=tk.W, fg="#0369a1", font=("Helvetica", 9, "italic"))
        status_lbl.pack(fill=tk.X, pady=(0, 5))

        # Bottom Area: Notebook with Tabs (Results & Visual Preview)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Formatted Results
        tab_res = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_res, text="  Analysis Result  ")

        self.txt_result = tk.Text(
            tab_res,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#1e293b",
            padx=10,
            pady=10,
            relief=tk.SOLID,
            borderwidth=1,
        )
        scroll_res = ttk.Scrollbar(tab_res, command=self.txt_result.yview)
        self.txt_result.configure(yscrollcommand=scroll_res.set)
        self.txt_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_res.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: Visual Preview (Chart / Page Rendering)
        self.tab_preview = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_preview, text="  Visual Preview  ")

        self.lbl_image_preview = tk.Label(self.tab_preview, text="No visual preview loaded.", bg="#f1f5f9")
        self.lbl_image_preview.pack(fill=tk.BOTH, expand=True)

    def _on_operation_change(self, event=None):
        op = self.op_var.get()
        if op == "Direct Query (Search Knowledge Base)":
            self.file_entry.configure(state="disabled")
            self.btn_browse.configure(state="disabled")
            self.query_entry.configure(state="normal")
            self.query_entry.focus_set()
        elif "Ask Question" in op:
            self.file_entry.configure(state="normal")
            self.btn_browse.configure(state="normal")
            self.query_entry.configure(state="normal")
        elif op == "System & Qdrant Inspection":
            self.file_entry.configure(state="disabled")
            self.btn_browse.configure(state="disabled")
            self.query_entry.configure(state="disabled")
        else:
            self.file_entry.configure(state="normal")
            self.btn_browse.configure(state="normal")
            self.query_entry.configure(state="disabled")

    def _browse_file(self):
        filetypes = [
            ("Supported Files", "*.pdf;*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff;*.csv;*.xlsx;*.txt;*.md"),
            ("PDF Documents (*.pdf)", "*.pdf"),
            ("Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff)", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff"),
            ("Data Tables (*.csv;*.xlsx;*.xls)", "*.csv;*.xlsx;*.xls"),
            ("All Files (*.*)", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            title="Select Target File",
            initialdir=str(settings.INPUTS_DIR if settings.INPUTS_DIR.exists() else settings.PROJECT_ROOT),
            filetypes=filetypes,
        )
        if chosen:
            self.file_var.set(chosen)
            self._load_visual_preview(chosen)

    def _load_visual_preview(self, file_path: str):
        try:
            p = Path(file_path)
            ft = detect_file_type(p)
            pil_img = None
            if ft == "image":
                pil_img = Image.open(str(p))
            elif ft == "pdf":
                import pymupdf

                doc = pymupdf.open(str(p))
                if len(doc) > 0:
                    pix = doc[0].get_pixmap(dpi=100)
                    import io

                    pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                doc.close()

            if pil_img:
                # Resize proportionally for preview container
                pil_img.thumbnail((650, 480), Image.Resampling.LANCZOS)
                self.current_preview_image = ImageTk.PhotoImage(pil_img)
                self.lbl_image_preview.configure(image=self.current_preview_image, text="")
            else:
                self.lbl_image_preview.configure(image="", text=f"File selected: {p.name} ({ft})")
        except Exception as exc:
            self.lbl_image_preview.configure(image="", text=f"Preview unavailable: {exc}")

    def _start_execution(self):
        op = self.op_var.get()
        file_path = self.file_var.get().strip()
        query = self.query_var.get().strip()

        # Validation
        if op != "Direct Query (Search Knowledge Base)" and op != "System & Qdrant Inspection":
            if not file_path:
                messagebox.showwarning("File Required", "Please browse or select a target file.")
                return
            if not Path(file_path).exists():
                messagebox.showerror("File Not Found", f"The file '{file_path}' does not exist.")
                return

        if "Ask Question" in op or op == "Direct Query (Search Knowledge Base)":
            if not query:
                messagebox.showwarning("Query Required", "Please enter your question.")
                return

        self.btn_run.configure(state="disabled", bg="#94a3b8", text="Processing...")
        self.status_var.set(f"Executing: {op}...")
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, f"Starting {op}...\nPlease wait while the AI agent processes the request...\n\n")

        # Run background thread
        threading.Thread(target=self._run_task_thread, args=(op, file_path, query), daemon=True).start()

    def _run_task_thread(self, op: str, file_path: str, query: str):
        try:
            if op == "Direct Query (Search Knowledge Base)":
                if not self.pipeline.role:
                    raise RuntimeError(
                        "No role configured. Set VISION_AGENT_ROLE (e.g. COO) "
                        "to query the enterprise corpus.")
                retrieved = self.pipeline.retriever.retrieve(
                    query, role=self.pipeline.role,
                    department=self.pipeline.department, top_k=4)
                context_str = "\n\n".join([f"[{c.file_name} (Page {c.page or 'N/A'})]: {c.text}" for c in retrieved])
                prompt = f"Answer the user query based strictly on this retrieved context:\n\n{context_str}\n\nQuery: {query}"
                ans = self.pipeline.vision.ollama.generate(prompt=prompt)

                out = (
                    f"============================================================\n"
                    f"DIRECT QUERY RESULT\n"
                    f"============================================================\n\n"
                    f"User Query:\n  {query}\n\n"
                    f"Answer:\n  {ans}\n\n"
                    f"Retrieved Sources:\n"
                )
                for c in retrieved:
                    out += f"  - Document: {c.file_name} | Page: {c.page or 'N/A'} | Dept: {c.department} (Score: {c.score:.3f})\n"

                self.root.after(0, self._render_results, out, None)

            elif op == "System & Qdrant Inspection":
                models = self.pipeline.vision.ollama.list_models()
                cols = self.pipeline.retriever.qdrant.list_collections()
                out = (
                    f"============================================================\n"
                    f"SYSTEM & DATABASE STATUS\n"
                    f"============================================================\n\n"
                    f"Ollama Server ({settings.OLLAMA_BASE_URL}): CONNECTED\n"
                    f"Installed Models: {', '.join(models)}\n"
                    f"Active Vision Model: {settings.VISION_MODEL}\n\n"
                    f"Qdrant Server ({settings.QDRANT_URL}): CONNECTED\n"
                    f"Collections Found: {len(cols)}\n"
                )
                for c in cols:
                    out += f"  - {c['name']}: {c['points_count']} points, dim: {c['dimension']}\n"
                self.root.after(0, self._render_results, out, None)

            else:
                cmd_map = {
                    "Summarize Document / Image": "summarize",
                    "Ask Question (Document VQA + Qdrant)": "ask",
                    "Data Visualization & Chart Generation": "visualize",
                    "CAD Blueprint & Diagram Analysis": "analyze",
                    "Extract Tables to CSV": "extract",
                    "Deep Multimodal Analysis": "analyze",
                }
                cmd = cmd_map.get(op, "analyze")
                result: PipelineResult = self.pipeline.run(file_path, command=cmd, question=query)

                out = (
                    f"============================================================\n"
                    f"VISION AGENT RESULT\n"
                    f"============================================================\n\n"
                    f"Input: {result.input_path.name}\n"
                    f"Detected Type: {result.detected_type}\n"
                    f"Pages Processed: {result.total_pages}\n\n"
                    f"Answer / Summary:\n{result.final_answer}\n\n"
                )

                if result.key_findings:
                    out += "Key Findings:\n" + "\n".join([f"  - {k}" for k in result.key_findings]) + "\n\n"

                if result.retrieved_chunks:
                    out += "Retrieved Context (from Qdrant):\n"
                    for c in result.retrieved_chunks:
                        out += f"  - [{c.file_name} (Page {c.page or 'N/A'}) Score: {c.score:.3f}]\n    \"{c.text.strip()[:140]}...\"\n"
                    out += "\n"

                if result.visual_findings:
                    out += f"Visual Observations:\n{result.visual_findings}\n\n"

                if result.generated_files:
                    out += "Generated Files:\n"
                    for gf in result.generated_files:
                        out += f"  - {gf}\n"

                # Check if a chart was generated to show in preview
                chart_file = next((f for f in result.generated_files if str(f).endswith(".png")), None)
                self.root.after(0, self._render_results, out, chart_file or file_path)

        except Exception as exc:
            self.root.after(0, self._render_error, str(exc))

    def _render_results(self, output_text: str, preview_file: Optional[str | Path]):
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, output_text)
        self.status_var.set("Completed successfully.")
        self.btn_run.configure(state="normal", bg="#0284c7", text="Run Operation")
        if preview_file:
            self._load_visual_preview(str(preview_file))

    def _render_error(self, error_message: str):
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, f"[ERROR]: {error_message}")
        self.status_var.set("Execution encountered an error.")
        self.btn_run.configure(state="normal", bg="#0284c7", text="Run Operation")
        messagebox.showerror("Error", error_message)


def launch_desktop_ui():
    """Entry point for native desktop UI."""
    root = tk.Tk()
    app = VisionAgentApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_desktop_ui()

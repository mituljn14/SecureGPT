"""
Script to generate all system diagrams and compile the complete, publication-grade
Vision Agent System Specification Document (PDF & HTML) with real embedded images.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "outputs" / "documentation_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_architecture_diagram() -> Path:
    """Generates an aesthetic, labeled system architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    # Title
    ax.text(0.5, 0.96, "SECURE VISION AGENT — PRODUCTION SYSTEM ARCHITECTURE", 
            fontsize=16, fontweight="bold", color="#f8fafc", ha="center", va="top")
    ax.text(0.5, 0.93, "End-to-End Multimodal Document & Vision Intelligence with Ollama & Qdrant", 
            fontsize=11, color="#94a3b8", ha="center", va="top")

    # Layer 1: User Interface
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.77), 0.90, 0.12, boxstyle="round,pad=0.015", 
                                        ec="#38bdf8", fc="#1e293b", lw=2))
    ax.text(0.08, 0.865, "1. USER INTERFACE & DISPATCH LAYER", fontsize=11, fontweight="bold", color="#38bdf8")
    
    ui_boxes = [
        ("CLI Entry (main.py)", 0.08, 0.79),
        ("Interactive Wizard (cli/interactive.py)", 0.32, 0.79),
        ("Desktop GUI Window (ui/desktop_app.py)", 0.61, 0.79),
        ("File Picker (Native OS)", 0.84, 0.79),
    ]
    for label, x, y in ui_boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), 0.11, 0.06, boxstyle="round,pad=0.01", ec="#0284c7", fc="#0369a1", lw=1))
        ax.text(x + 0.055, y + 0.03, label, fontsize=8, color="#ffffff", ha="center", va="center", weight="bold", wrap=True)

    # Layer 2: Core Orchestration
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.60), 0.90, 0.12, boxstyle="round,pad=0.015", 
                                        ec="#818cf8", fc="#1e1b4b", lw=2))
    ax.text(0.08, 0.695, "2. CORE INGESTION & PIPELINE ORCHESTRATION", fontsize=11, fontweight="bold", color="#818cf8")
    
    core_boxes = [
        ("File Type Detector & Security Validator (utils/file_utils.py)", 0.08, 0.62, 0.26),
        ("Intelligent Task Router (core/router.py)", 0.38, 0.62, 0.24),
        ("Vision Pipeline Coordinator (core/pipeline.py)", 0.66, 0.62, 0.27),
    ]
    for label, x, y, w in core_boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, 0.06, boxstyle="round,pad=0.01", ec="#6366f1", fc="#4338ca", lw=1))
        ax.text(x + w/2, y + 0.03, label, fontsize=8, color="#ffffff", ha="center", va="center", weight="bold")

    # Layer 3: Extraction & Vision Engines
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.41), 0.90, 0.15, boxstyle="round,pad=0.015", 
                                        ec="#a855f7", fc="#2e1065", lw=2))
    ax.text(0.08, 0.535, "3. MULTIMODAL EXTRACTION & VISION SUBSYSTEMS", fontsize=11, fontweight="bold", color="#a855f7")

    sub_boxes = [
        ("PDF Engine & Pixmap Renderer\n(document/pdf_processor.py)", 0.08, 0.43, 0.20),
        ("RapidOCR ONNX Engine\n(document/ocr_processor.py)", 0.31, 0.43, 0.18),
        ("pdfplumber Table Parser\n(document/table_extractor.py)", 0.52, 0.43, 0.19),
        ("Ollama Vision LLM Gateway\n(Moondream / Qwen-VL / Llama3.2)", 0.74, 0.43, 0.20),
    ]
    for label, x, y, w in sub_boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, 0.08, boxstyle="round,pad=0.01", ec="#c084fc", fc="#7e22ce", lw=1))
        ax.text(x + w/2, y + 0.04, label, fontsize=8, color="#ffffff", ha="center", va="center", weight="bold")

    # Layer 4: Analytics & Knowledge Retrieval
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.21), 0.90, 0.16, boxstyle="round,pad=0.015", 
                                        ec="#34d399", fc="#064e3b", lw=2))
    ax.text(0.08, 0.345, "4. ANALYTICS & QDRANT KNOWLEDGE RETRIEVAL LAYER", fontsize=11, fontweight="bold", color="#34d399")

    ret_boxes = [
        ("BAAI/bge-m3 Embeddings\n(1024-dimension vectors)", 0.08, 0.23, 0.20),
        ("Qdrant Vector Database\n(344,458 vectors @ Cosine)", 0.31, 0.23, 0.21),
        ("Pandas & NumPy Analytics\n(Totals, Means, Min/Max, Trends)", 0.55, 0.23, 0.21),
        ("Matplotlib & Seaborn\n(Automated Bar/Line/Pie Plots)", 0.78, 0.23, 0.16),
    ]
    for label, x, y, w in ret_boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, 0.09, boxstyle="round,pad=0.01", ec="#10b981", fc="#047857", lw=1))
        ax.text(x + w/2, y + 0.045, label, fontsize=8, color="#ffffff", ha="center", va="center", weight="bold")

    # Layer 5: Output & Artifacts
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.03), 0.90, 0.14, boxstyle="round,pad=0.015", 
                                        ec="#fb923c", fc="#451a03", lw=2))
    ax.text(0.08, 0.145, "5. OUTPUT DISPATCH & PERSISTENT ARTIFACTS", fontsize=11, fontweight="bold", color="#fb923c")

    out_boxes = [
        ("Tee Dual Logger\n(Terminal + logs/*.log)", 0.08, 0.05, 0.20),
        ("Structured Summaries\n(outputs/summaries/*.md)", 0.31, 0.05, 0.21),
        ("Extracted Tables\n(outputs/extracted/*.csv)", 0.55, 0.05, 0.20),
        ("Saved Charts\n(outputs/visualizations/*.png)", 0.78, 0.05, 0.16),
    ]
    for label, x, y, w in out_boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, 0.075, boxstyle="round,pad=0.01", ec="#f97316", fc="#c2410c", lw=1))
        ax.text(x + w/2, y + 0.037, label, fontsize=8, color="#ffffff", ha="center", va="center", weight="bold")

    # Connector Arrows
    arrow_props = dict(arrowstyle="->", color="#38bdf8", lw=2.5)
    ax.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.77), arrowprops=arrow_props)
    ax.annotate("", xy=(0.5, 0.56), xytext=(0.5, 0.60), arrowprops=dict(arrowstyle="->", color="#818cf8", lw=2.5))
    ax.annotate("", xy=(0.5, 0.37), xytext=(0.5, 0.41), arrowprops=dict(arrowstyle="->", color="#a855f7", lw=2.5))
    ax.annotate("", xy=(0.5, 0.17), xytext=(0.5, 0.21), arrowprops=dict(arrowstyle="->", color="#34d399", lw=2.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()

    out_path = ASSETS_DIR / "system_architecture_diagram.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return out_path


def generate_dataflow_diagram() -> Path:
    """Generates the step-by-step pipeline dataflow diagram."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#1e293b")

    ax.text(0.5, 0.92, "PIPELINE DATA FLOW & TRANSFORMATION SEQUENCE", 
            fontsize=14, fontweight="bold", color="#f8fafc", ha="center")

    steps = [
        ("Step 1: Ingest", "Input File\n(PDF/IMG/CAD/CSV)", "#0284c7"),
        ("Step 2: Parse", "PyMuPDF Text +\nRapidOCR ONNX", "#4f46e5"),
        ("Step 3: Vision", "Ollama Moondream\nVisual Analysis", "#7c3aed"),
        ("Step 4: Retrieve", "Qdrant Vector DB\n(1024d BGE-M3)", "#059669"),
        ("Step 5: Compute", "Pandas Analytics +\nSeaborn Plotting", "#d97706"),
        ("Step 6: Synthesize", "Grounded Response\n+ Saved Artifacts", "#dc2626"),
    ]

    x_positions = np.linspace(0.06, 0.82, len(steps))
    for i, (title, desc, color) in enumerate(steps):
        x = x_positions[i]
        # Step Header
        ax.text(x + 0.05, 0.72, title, fontsize=9, fontweight="bold", color="#94a3b8", ha="center")
        # Step Box
        ax.add_patch(patches.FancyBboxPatch((x, 0.25), 0.10, 0.40, boxstyle="round,pad=0.015", 
                                            ec=color, fc="#0f172a", lw=2))
        ax.text(x + 0.05, 0.45, desc, fontsize=8, color="#f8fafc", ha="center", va="center", weight="bold")

        # Arrow to next step
        if i < len(steps) - 1:
            ax.annotate("", xy=(x_positions[i+1] - 0.01, 0.45), xytext=(x + 0.11, 0.45),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()

    out_path = ASSETS_DIR / "dataflow_sequence_diagram.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return out_path


def generate_cad_rendered_image() -> Path:
    """Renders the CAD blueprint PDF into a high-res PNG."""
    cad_pdf_path = PROJECT_ROOT / "inputs" / "cad_engineering_blueprint.pdf"
    doc = pymupdf.open(str(cad_pdf_path))
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    out_path = ASSETS_DIR / "cad_blueprint_drawing.png"
    pix.save(str(out_path))
    doc.close()
    return out_path


def generate_seaborn_sample_chart() -> Path:
    """Generates a publication-grade Seaborn bar chart for inclusion in the doc."""
    import pandas as pd
    df = pd.DataFrame({
        "Region": ["North", "South", "East", "West"],
        "Revenue ($)": [45000, 52000, 31000, 68000],
        "Growth (%)": [12.5, 8.2, -3.1, 19.4],
    })
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df, x="Region", y="Revenue ($)", hue="Region", palette="viridis", legend=False, ax=ax)
    ax.set_title("Regional Revenue Performance Breakdown", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Revenue in USD ($)", fontsize=10)
    ax.set_xlabel("Operational Region", fontsize=10)
    for p in ax.patches:
        ax.annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 7), textcoords="offset points", fontsize=9, fontweight="bold")
    plt.tight_layout()
    out_path = ASSETS_DIR / "sample_seaborn_chart.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def generate_desktop_gui_mockup() -> Path:
    """Generates an illustrative render of the Tkinter Desktop GUI interface."""
    img = Image.new("RGB", (1000, 650), color="#f8fafc")
    draw = ImageDraw.Draw(img)

    # Top Header
    draw.rectangle([0, 0, 1000, 60], fill="#1e293b")
    draw.text((25, 20), "Vision Agent — Multimodal & Document Intelligence", fill="#f8fafc")
    draw.text((680, 22), "Vision: moondream | Qdrant: securegpt_corpus", fill="#94a3b8")

    # Control Frame
    draw.rectangle([20, 75, 980, 180], outline="#cbd5e1", fill="#ffffff", width=2)
    draw.text((35, 68), " Task Controls & Input ", fill="#0284c7")

    draw.text((35, 100), "Select Operation:", fill="#1e293b")
    draw.rectangle([160, 95, 450, 125], outline="#94a3b8", fill="#f1f5f9")
    draw.text((170, 103), "CAD Blueprint & Diagram Analysis", fill="#0f172a")

    draw.text((470, 100), "Target File:", fill="#1e293b")
    draw.rectangle([555, 95, 850, 125], outline="#94a3b8", fill="#f1f5f9")
    draw.text((565, 103), "inputs/cad_engineering_blueprint.pdf", fill="#0f172a")

    draw.rectangle([865, 95, 960, 125], outline="#0284c7", fill="#0284c7")
    draw.text((885, 103), "Browse...", fill="#ffffff")

    draw.text((35, 140), "Question / Query:", fill="#1e293b")
    draw.rectangle([160, 135, 850, 165], outline="#94a3b8", fill="#ffffff")
    draw.text((170, 143), "What is the pressure rating and bolt torque specification?", fill="#334155")

    draw.rectangle([865, 135, 960, 165], outline="#059669", fill="#059669")
    draw.text((880, 143), "Run Operation", fill="#ffffff")

    # Status
    draw.text((25, 192), "Status: Completed successfully in 1.4s.", fill="#0369a1")

    # Results Notebook (Left: text, Right: visual preview)
    draw.rectangle([20, 215, 540, 630], outline="#cbd5e1", fill="#ffffff", width=1)
    draw.rectangle([20, 215, 180, 240], fill="#0284c7")
    draw.text((30, 222), "Analysis Result", fill="#ffffff")

    res_sample = (
        "=== CAD BLUEPRINT ANALYSIS RESULT ===\n\n"
        "Part Name: High-Pressure Flange Assembly\n"
        "Drawing No: DWG-ENG-2026-4401 (Rev D)\n"
        "Material: 316L Stainless Steel\n"
        "Pressure Rating: Class 600 (1480 PSI)\n"
        "Hydrostatic Test: 2220 PSI for 30 min\n\n"
        "Bill of Materials (BOM):\n"
        "- 2x Flange Body Weld Neck (4.0 in)\n"
        "- 1x Spiral Wound Gasket (Graphite/316L)\n"
        "- 8x Stud Bolts (7/8 in x 5.5 in)\n"
        "- 16x Heavy Hex Nuts (ASTM A194 Gr 8M)\n\n"
        "Qdrant Grounded Citations:\n"
        "- CAD_Part_Release_Log.csv (Score: 0.585)"
    )
    draw.text((35, 255), res_sample, fill="#0f172a")

    # Visual Preview Tab (Right side)
    draw.rectangle([560, 215, 980, 630], outline="#cbd5e1", fill="#f8fafc", width=1)
    draw.rectangle([560, 215, 700, 240], fill="#0284c7")
    draw.text((570, 222), "Visual Preview", fill="#ffffff")

    # Paste thumbnail of CAD blueprint
    cad_thumb = Image.open(ASSETS_DIR / "cad_blueprint_drawing.png")
    cad_thumb.thumbnail((400, 360), Image.Resampling.LANCZOS)
    img.paste(cad_thumb, (570, 250))

    out_path = ASSETS_DIR / "desktop_gui_mockup.png"
    img.save(str(out_path))
    return out_path


def compile_complete_pdf_specification(
    arch_img: Path,
    flow_img: Path,
    cad_img: Path,
    chart_img: Path,
    gui_img: Path,
) -> Path:
    """Compiles a professional, multi-page PDF document with all diagrams and sample images."""
    doc = pymupdf.open()

    # ----------------------------------------------------
    # PAGE 1: Title & Executive Architecture Specification
    # ----------------------------------------------------
    p1 = doc.new_page(width=612, height=792)  # US Letter Portrait
    p1.draw_rect(pymupdf.Rect(20, 20, 592, 772), color=(0.1, 0.15, 0.2), width=1.5)

    # Header Banner
    p1.draw_rect(pymupdf.Rect(22, 22, 590, 85), color=(0.08, 0.12, 0.2), fill=(0.08, 0.12, 0.2))
    p1.insert_text((35, 52), "LOCAL VISION AGENT — SYSTEM SPECIFICATION", fontsize=15, fontname="helv", color=(1, 1, 1))
    p1.insert_text((35, 72), "Production Technical Manual, Multimodal Architecture & Workflow Reference", fontsize=9, fontname="helv", color=(0.6, 0.75, 0.9))

    p1.insert_text((35, 110), "1. SYSTEM ARCHITECTURE & COMPONENT TOPOLOGY", fontsize=11, fontname="helv", color=(0.05, 0.2, 0.4))
    p1.insert_text((35, 125), "The Local Vision Agent integrates 5 distinct layers: User Interface, Core Pipeline Router,", fontsize=9, fontname="helv")
    p1.insert_text((35, 138), "Document/OCR/Vision Extractors, Qdrant Vector Retrieval (1024d), and Automated Analytics/Plotting.", fontsize=9, fontname="helv")

    # Embed Architecture Diagram
    p1.insert_image(pymupdf.Rect(35, 150, 577, 490), filename=str(arch_img))

    p1.insert_text((35, 505), "Key Subsystem Highlights:", fontsize=9.5, fontname="helv", color=(0.05, 0.2, 0.4))
    highlights = [
        "- Ollama Vision LLM Gateway: Zero cloud dependencies, runs Moondream / Qwen2.5-VL with 1280px resolution capping.",
        "- Qdrant Vector Database: Connects to existing 344,458 vector chunks @ 1024-dim BAAI/bge-m3 cosine similarity.",
        "- Document & OCR Subsystem: PyMuPDF 150 DPI page rendering + standalone RapidOCR ONNX engine.",
        "- Automated Analytics: Pandas statistical engine & Seaborn publication-grade bar/line/pie chart generation.",
        "- Tee Dual Logging: Real-time duplication of terminal standard output/errors to timestamped logs/ files."
    ]
    hy = 522
    for h in highlights:
        p1.insert_text((40, hy), h, fontsize=8, fontname="helv")
        hy += 15

    # Footer
    p1.insert_text((35, 760), "SecureGPT Local Vision Agent Specification | Page 1", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    # ----------------------------------------------------
    # PAGE 2: Dataflow Sequence & CAD Blueprint Analysis
    # ----------------------------------------------------
    p2 = doc.new_page(width=612, height=792)
    p2.draw_rect(pymupdf.Rect(20, 20, 592, 772), color=(0.1, 0.15, 0.2), width=1.5)

    p2.insert_text((35, 45), "2. PIPELINE DATA FLOW & EXECUTION SEQUENCE", fontsize=11, fontname="helv", color=(0.05, 0.2, 0.4))
    p2.insert_text((35, 60), "Sequence of data transformation from raw multi-format input to grounded multimodal artifacts:", fontsize=9, fontname="helv")

    # Embed Dataflow Diagram
    p2.insert_image(pymupdf.Rect(35, 70, 577, 260), filename=str(flow_img))

    p2.insert_text((35, 280), "3. SAMPLE 1: CAD BLUEPRINT & TECHNICAL DIAGRAM ANALYSIS", fontsize=11, fontname="helv", color=(0.05, 0.2, 0.4))
    p2.insert_text((35, 295), "Ingestion of mechanical engineering drawings, title-block parsing, Bill of Materials, and specs:", fontsize=9, fontname="helv")

    # Embed CAD Blueprint Drawing
    p2.insert_image(pymupdf.Rect(35, 308, 330, 530), filename=str(cad_img))

    # CAD Output Box on Right
    p2.draw_rect(pymupdf.Rect(340, 308, 577, 530), color=(0.8, 0.85, 0.9), fill=(0.95, 0.97, 1.0))
    p2.insert_text((350, 325), "EXTRACTED CAD SPECIFICATIONS:", fontsize=9, fontname="helv", color=(0.1, 0.2, 0.5))
    
    cad_text_lines = [
        "Part: High-Pressure Flange Assy",
        "Drawing No: DWG-ENG-2026-4401 (Rev D)",
        "Material: 316L Stainless Steel",
        "Pressure Rating: Class 600 (1480 PSI)",
        "Test: Hydrostatic @ 2220 PSI for 30m",
        "",
        "Bill of Materials (BOM Table):",
        "- Item 1: 2x Flange Body (4in, 316L)",
        "- Item 2: 1x Spiral Gasket (Graphite)",
        "- Item 3: 8x Stud Bolts (7/8in x 5.5in)",
        "- Item 4: 16x Heavy Hex Nuts (Gr 8M)",
        "",
        "Dimensions & Callouts:",
        "- Inner Bore: 4.00 in (101.6 mm)",
        "- Bolt Circle: 8.50 in (8x Holes)",
        "- Outer Diameter: 10.75 in",
        "",
        "Qdrant Citations:",
        "- CAD_Part_Release_Log.csv",
        "- Automotive_Engineering_Catalog.csv"
    ]
    ty = 342
    for line in cad_text_lines:
        p2.insert_text((350, ty), line, fontsize=7.5, fontname="helv")
        ty += 11.5

    # Footer
    p2.insert_text((35, 760), "SecureGPT Local Vision Agent Specification | Page 2", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    # ----------------------------------------------------
    # PAGE 3: Tabular Analytics, Seaborn Charts & Desktop UI
    # ----------------------------------------------------
    p3 = doc.new_page(width=612, height=792)
    p3.draw_rect(pymupdf.Rect(20, 20, 592, 772), color=(0.1, 0.15, 0.2), width=1.5)

    p3.insert_text((35, 45), "4. SAMPLE 2: TABULAR ANALYTICS & SEABORN VISUALIZATION", fontsize=11, fontname="helv", color=(0.05, 0.2, 0.4))
    p3.insert_text((35, 60), "Automatic conversion of extracted tables into Pandas DataFrames, statistical summaries & plots:", fontsize=9, fontname="helv")

    # Embed Seaborn Sample Chart
    p3.insert_image(pymupdf.Rect(35, 72, 330, 275), filename=str(chart_img))

    # Tabular Output Box on Right
    p3.draw_rect(pymupdf.Rect(340, 72, 577, 275), color=(0.8, 0.85, 0.9), fill=(0.95, 0.97, 1.0))
    p3.insert_text((350, 90), "COMPUTED STATISTICAL INSIGHTS:", fontsize=9, fontname="helv", color=(0.1, 0.2, 0.5))
    
    stats_lines = [
        "Input: sample_sales.csv (4 rows, 3 cols)",
        "",
        "Revenue Analytics (Pandas):",
        "- Total Revenue: $196,000.00",
        "- Mean Revenue: $49,000.00",
        "- Min Revenue: $31,000.00 (East)",
        "- Max Revenue: $68,000.00 (West)",
        "- Growth Trend: +51.11% (North->West)",
        "",
        "Generated Artifacts:",
        "- outputs/extracted/sample_sales_table.csv",
        "- outputs/visualizations/bar_chart_*.png",
        "- outputs/summaries/sample_sales_summary.md"
    ]
    sy = 108
    for line in stats_lines:
        p3.insert_text((350, sy), line, fontsize=8, fontname="helv")
        sy += 13.5

    p3.insert_text((35, 290), "5. DUAL INTERFACES: INTERACTIVE TERMINAL WIZARD & NATIVE DESKTOP GUI", fontsize=11, fontname="helv", color=(0.05, 0.2, 0.4))
    p3.insert_text((35, 305), "Provides both a rich terminal wizard and a zero-server Tkinter desktop window with live preview:", fontsize=9, fontname="helv")

    # Embed Desktop GUI Mockup
    p3.insert_image(pymupdf.Rect(35, 318, 577, 650), filename=str(gui_img))

    p3.insert_text((35, 665), "Interface Features:", fontsize=9.5, fontname="helv", color=(0.05, 0.2, 0.4))
    gui_features = [
        "- Task Selector: Instant switching between Summarize, VQA, Direct Qdrant Query, Visualization, CAD & Tables.",
        "- Native Windows Explorer Picker: Graphical file selection with type filtering (*.pdf, *.png, *.csv, etc.).",
        "- Threaded Non-Blocking Execution: Keeps GUI responsive while Ollama and Qdrant execute in background.",
        "- Live Dual-Tab Results: Structured formatted markdown on Tab 1, and live visual preview/chart rendering on Tab 2."
    ]
    gy = 682
    for gf in gui_features:
        p3.insert_text((40, gy), gf, fontsize=8, fontname="helv")
        gy += 14

    # Footer
    p3.insert_text((35, 760), "SecureGPT Local Vision Agent Specification | Page 3", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    pdf_out_path = PROJECT_ROOT / "VISION_AGENT_SYSTEM_SPECIFICATION.pdf"
    doc.save(str(pdf_out_path))
    doc.close()
    return pdf_out_path


def compile_interactive_html_specification(
    arch_img: Path,
    flow_img: Path,
    cad_img: Path,
    chart_img: Path,
    gui_img: Path,
) -> Path:
    """Generates a standalone, beautifully styled HTML document with embedded images."""
    import base64

    def to_b64(img_path: Path) -> str:
        return base64.b64encode(img_path.read_bytes()).decode("utf-8")

    arch_b64 = to_b64(arch_img)
    flow_b64 = to_b64(flow_img)
    cad_b64 = to_b64(cad_img)
    chart_b64 = to_b64(chart_img)
    gui_b64 = to_b64(gui_img)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Local Vision Agent — Production Architecture & Technical Specification</title>
<style>
  :root {{
    --bg: #0f172a;
    --card-bg: #1e293b;
    --text: #f8fafc;
    --muted: #94a3b8;
    --primary: #38bdf8;
    --secondary: #818cf8;
    --accent: #34d399;
    --border: #334155;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
    line-height: 1.6;
  }}
  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 2px solid var(--border);
    padding: 40px 20px;
    text-align: center;
  }}
  .header h1 {{
    margin: 0;
    font-size: 2.2rem;
    color: var(--primary);
  }}
  .header p {{
    color: var(--muted);
    margin-top: 8px;
    font-size: 1.1rem;
  }}
  .container {{
    max-width: 1100px;
    margin: 30px auto;
    padding: 0 20px;
  }}
  .section {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 30px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }}
  h2 {{
    color: var(--primary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-top: 0;
  }}
  .img-container {{
    text-align: center;
    margin: 20px 0;
  }}
  .img-container img {{
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: center;
  }}
  .card {{
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    font-size: 0.95rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
  }}
  th, td {{
    padding: 10px 14px;
    border: 1px solid var(--border);
    text-align: left;
    font-size: 0.9rem;
  }}
  th {{
    background: #0f172a;
    color: var(--primary);
  }}
  .badge {{
    background: #0284c7;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: bold;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Local Vision Agent — Production Technical Specification</h1>
  <p>Local Multimodal Intelligence Engine • Ollama Vision LLMs • 344k Qdrant Vectors • Pandas & Seaborn</p>
</div>

<div class="container">

  <!-- Section 1: Architecture -->
  <div class="section">
    <h2>1. System Architecture & Component Topology</h2>
    <p>The system provides a 5-layer modular architecture connecting dual user interfaces (CLI & Desktop GUI) to local Ollama vision models and a local Qdrant vector database:</p>
    <div class="img-container">
      <img src="data:image/png;base64,{arch_b64}" alt="System Architecture Diagram">
    </div>
  </div>

  <!-- Section 2: Data Flow -->
  <div class="section">
    <h2>2. Pipeline Data Flow & Execution Sequence</h2>
    <p>Transformation sequence handling input detection, native text & OCR extraction, visual understanding, Qdrant retrieval, tabular analytics, and artifact generation:</p>
    <div class="img-container">
      <img src="data:image/png;base64,{flow_b64}" alt="Pipeline Dataflow Sequence">
    </div>
  </div>

  <!-- Section 3: CAD Analysis Sample -->
  <div class="section">
    <h2>3. Sample 1: CAD Blueprint & Diagram Ingestion</h2>
    <div class="grid-2">
      <div class="img-container">
        <img src="data:image/png;base64,{cad_b64}" alt="CAD Blueprint Drawing">
      </div>
      <div class="card">
        <h3 style="color: var(--accent); margin-top:0;">Extracted CAD Blueprint Specs:</h3>
        <p><strong>Part Name:</strong> High-Pressure Flange Assembly<br>
        <strong>Drawing No:</strong> DWG-ENG-2026-4401 (Rev D)<br>
        <strong>Material:</strong> 316L Stainless Steel<br>
        <strong>Pressure Rating:</strong> Class 600 (1480 PSI)<br>
        <strong>Hydrostatic Test:</strong> 2220 PSI for 30 minutes</p>
        <p><strong>Bill of Materials (BOM):</strong><br>
        • 2x Flange Body Weld Neck 4.0"<br>
        • 1x Spiral Wound Gasket (Graphite/316L)<br>
        • 8x Stud Bolts (7/8" x 5.5")<br>
        • 16x Heavy Hex Nuts (ASTM A194 Gr 8M)</p>
        <p><strong>Grounded Qdrant Citations:</strong><br>
        • <code>CAD_Part_Release_Log.csv</code> (Score: 0.585)<br>
        • <code>Automotive_Engineering_Catalog.csv</code> (Score: 0.574)</p>
      </div>
    </div>
  </div>

  <!-- Section 4: Data Visualization Sample -->
  <div class="section">
    <h2>4. Sample 2: Tabular Analytics & Seaborn Visualization</h2>
    <div class="grid-2">
      <div class="img-container">
        <img src="data:image/png;base64,{chart_b64}" alt="Seaborn Bar Plot">
      </div>
      <div class="card">
        <h3 style="color: var(--accent); margin-top:0;">Computed Statistical Insights:</h3>
        <p><strong>Input:</strong> <code>sample_sales.csv</code> (4 rows, 3 columns)</p>
        <p><strong>Revenue Analytics (Pandas):</strong><br>
        • Total Revenue: <strong>$196,000.00</strong><br>
        • Mean Revenue: <strong>$49,000.00</strong><br>
        • Min Revenue: <strong>$31,000.00</strong> (East Region)<br>
        • Max Revenue: <strong>$68,000.00</strong> (West Region)<br>
        • Multi-Period Trend: <strong>+51.11% growth</strong></p>
        <p><strong>Generated Files:</strong><br>
        • <code>outputs/extracted/sample_sales_table.csv</code><br>
        • <code>outputs/visualizations/bar_chart_*.png</code><br>
        • <code>outputs/summaries/sample_sales_summary.md</code></p>
      </div>
    </div>
  </div>

  <!-- Section 5: Native Desktop GUI -->
  <div class="section">
    <h2>5. Native Desktop GUI Application (Tkinter & Pillow)</h2>
    <p>A native desktop window with task selector, Windows File Explorer browse button, question input, and live dual-tab visual preview:</p>
    <div class="img-container">
      <img src="data:image/png;base64,{gui_b64}" alt="Desktop GUI Window">
    </div>
  </div>

</div>
</body>
</html>
"""
    html_out_path = PROJECT_ROOT / "VISION_AGENT_SYSTEM_SPECIFICATION.html"
    html_out_path.write_text(html_content, encoding="utf-8")
    return html_out_path


def main():
    print("Generating system architecture diagram...")
    arch_img = generate_architecture_diagram()
    print(f"[OK] Saved: {arch_img.name}")

    print("Generating dataflow sequence diagram...")
    flow_img = generate_dataflow_diagram()
    print(f"[OK] Saved: {flow_img.name}")

    print("Rendering CAD blueprint asset...")
    cad_img = generate_cad_rendered_image()
    print(f"[OK] Saved: {cad_img.name}")

    print("Generating Seaborn sample chart asset...")
    chart_img = generate_seaborn_sample_chart()
    print(f"[OK] Saved: {chart_img.name}")

    print("Generating Desktop GUI mockup asset...")
    gui_img = generate_desktop_gui_mockup()
    print(f"[OK] Saved: {gui_img.name}")

    print("Compiling publication-grade PDF specification...")
    pdf_path = compile_complete_pdf_specification(arch_img, flow_img, cad_img, chart_img, gui_img)
    print(f"[OK] Successfully created PDF document: {pdf_path}")

    print("Compiling interactive HTML specification...")
    html_path = compile_interactive_html_specification(arch_img, flow_img, cad_img, chart_img, gui_img)
    print(f"[OK] Successfully created HTML document: {html_path}")


if __name__ == "__main__":
    main()

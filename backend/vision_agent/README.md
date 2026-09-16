# Local Vision Agent

A production-quality local Vision & Document Intelligence Agent. The agent accepts images, native PDFs, scanned PDFs, screenshots, tables, charts, graphs, and diagrams. It extracts text via native extraction and OCR, understands visuals with local Vision LLMs via Ollama, integrates with an existing Qdrant vector database for grounded knowledge retrieval, performs tabular analysis with Pandas, and generates automated visualizations with Matplotlib and Seaborn.

---

## Architecture Overview

```
USER / TERMINAL
       │
       ▼
 INPUT DETECTION (PDF, Images, CAD visuals, Tables)
       │
       ▼
 PREPROCESSING & RESOLUTION LIMITS
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
 PDF / TEXT PARSER                 IMAGE PROCESSING
       │                                 │
       ▼                                 ▼
 OCR / EXTRACTION                  VISION INFERENCE
 (RapidOCR + fallback)              (Ollama: Moondream / Qwen2.5-VL)
       │                                 │
       └────────────────┬────────────────┘
                        │
                        ▼
                 TASK ROUTER & ANALYSIS
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   QDRANT RETRIEVAL             DATA ANALYSIS
 (Existing 1024-dim corpus)      (Pandas & NumPy)
          │                           │
          │                     VISUALIZATION
          │                     (Matplotlib / Seaborn)
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
             GROUNDED RESPONSE SYNTHESIS
                        │
                        ▼
              STRUCTURED CLI OUTPUT
```

---

## Folder Structure

```
vision_agent/
├── main.py                     # CLI entry point with Tee-logging and argument parsing
├── requirements.txt            # Minimal required dependencies
├── .env.example                # Example environment variables
├── .env                        # Local active configuration
├── README.md                   # Comprehensive system documentation
├── .gitignore                  # Git exclusions
│
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py             # Typed settings from environment
│
├── core/                       # Orchestration & Pipeline
│   ├── __init__.py
│   ├── models.py               # Domain models (OCRResult, TableData, ProcessedPage, PipelineResult)
│   ├── router.py               # Task router (SUMMARY, QA, OCR, TABLE, VISUALIZE)
│   └── pipeline.py             # Central pipeline orchestrating all components
│
├── vision/                     # Vision LLM & Image Understanding
│   ├── __init__.py
│   ├── ollama_client.py        # HTTP client for Ollama API
│   ├── image_processor.py      # Resizing, scaling, and Base64 conversion
│   ├── prompts.py              # Specialized vision and reasoning prompts
│   └── vision_agent.py         # High-level visual reasoning interface
│
├── document/                   # Document parsing & OCR
│   ├── __init__.py
│   ├── pdf_processor.py        # Multi-page PDF renderer, page counter, text extractor
│   ├── text_extractor.py       # Native selectable text extraction
│   ├── ocr_processor.py        # RapidOCR (ONNX) + pytesseract fallback
│   ├── table_extractor.py      # Native PDF + tabular file table extraction
│   └── document_analyzer.py    # Unified document entry point
│
├── retrieval/                  # Qdrant Vector Retrieval
│   ├── __init__.py
│   ├── qdrant_client.py        # Non-destructive read-only Qdrant client
│   ├── embeddings.py           # 1024-dim BGE-M3 embedding generator
│   └── retriever.py            # Similarity search & metadata extraction
│
├── data/                       # Structured Data Analysis
│   ├── __init__.py
│   ├── dataframe_processor.py  # Tabular cleaner, type inferrer, and summarizer
│   └── data_analyzer.py        # Column comparisons, totals, averages, and trend detector
│
├── visualization/              # Data Visualization
│   ├── __init__.py
│   └── chart_generator.py      # Safe template-driven plotting (Bar, Line, Pie)
│
├── cli/                        # Terminal Interface
│   ├── __init__.py
│   └── commands.py             # Command handlers & structured terminal formatter
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── file_utils.py           # File type detection, validation, path safety
│   ├── image_utils.py          # PIL utilities, format conversion, temporary storage
│   └── logging_utils.py        # Dual Terminal + Session File Tee-logger
│
├── tests/                      # Automated Verification Test Suite
│   ├── test_environment.py     # Verifies Python, Ollama, and Qdrant
│   ├── test_ollama.py          # Verifies Ollama text and image inference
│   ├── test_qdrant.py          # Verifies Qdrant collections and vector search
│   ├── test_ocr.py             # Verifies RapidOCR / OCR extraction
│   ├── test_pdf.py             # Verifies PDF text extraction & rendering
│   ├── test_vision.py          # Verifies visual reasoning on charts & diagrams
│   └── test_pipeline.py        # Verifies end-to-end processing pipeline
│
├── inputs/                     # Directory for user input files
├── outputs/                    # Output artifacts
│   ├── summaries/              # Generated markdown summaries
│   ├── extracted/              # Extracted tables (.csv)
│   ├── visualizations/         # Generated plots and charts (.png)
│   └── temporary/              # Intermediate rendered page images
│
└── logs/                       # Timestamped session logs
```

---

## Installation & Setup

### 1. Install Dependencies
```bash
cd vision_agent
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` (or configure via environment variables):
```ini
OLLAMA_BASE_URL=http://localhost:11434
VISION_MODEL=moondream
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=securegpt_corpus
EMBEDDING_MODEL=BAAI/bge-m3
MAX_IMAGE_DIMENSION=1280
LOG_LEVEL=INFO
```

### 3. Model & Service Setup

#### Ollama Vision Model
Ensure Ollama is running and pull the vision model:
```bash
ollama pull moondream
```
*(Or if using Qwen2.5-VL: `ollama pull qwen2.5-vl:3b`)*

#### Existing Qdrant Database
The retriever connects directly to your existing Qdrant instance on `localhost:6333` and uses the existing `securegpt_corpus` collection. It performs strictly read-only search operations and never alters or deletes existing vector data.

---

## CLI Usage

### 1. Summarize Document or Image
```bash
python main.py summarize <path_to_file>
```
*Processes the document, runs OCR if scanned, inspects visual components, and produces a structured summary.*

### 2. Ask Questions (Document VQA + Grounded Qdrant Retrieval)
```bash
python main.py ask <path_to_file> "What are the main quarterly revenue drivers?"
```
*Combines visual observations, document text, and relevant chunks retrieved from Qdrant to give a grounded answer.*

### 3. Deep Multimodal Analysis
```bash
python main.py analyze <path_to_file>
```
*Extracts tables, analyzes trends, explains diagrams or charts, and reports key insights.*

### 4. Structured Data & Table Extraction
```bash
python main.py extract <path_to_file>
```
*Extracts tables as clean CSV files into `outputs/extracted/` and formats markdown tables in terminal.*

### 5. Automatic Data Visualization
```bash
python main.py visualize <path_to_file>
```
*Extracts numerical data and generates bar/line/pie charts into `outputs/visualizations/`.*

### 6. System & Environment Inspection
```bash
python main.py inspect
```
*Inspects Ollama connectivity, model availability, and existing Qdrant collections and vector dimensions.*

---

## Running Verification Tests

Run the full automated test suite using `pytest` or directly with Python:

```bash
# Run all tests
pytest tests/ -v

# Or run individual test modules
python tests/test_environment.py
python tests/test_ollama.py
python tests/test_qdrant.py
python tests/test_ocr.py
python tests/test_pdf.py
python tests/test_vision.py
python tests/test_pipeline.py
```

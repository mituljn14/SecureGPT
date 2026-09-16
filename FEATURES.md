# SecureGPT - Architecture & Features

This document tracks the current architecture, pipelines, and features of the SecureGPT project.

## 🏗 Architecture & Pipelines

### 1. The Multi-Agent Supervisor Pattern (LangGraph)
- **Supervisor Node**: The entry point for all chat requests. It analyzes the user's prompt and routes it to the most capable specialized agent based on the query's intent (e.g., coding, operations, security, etc.).
- **Worker Agents**:
  - `operations`, `security`, `networking`, `identity`: RAG-enabled agents equipped to answer specific domain questions by querying the Qdrant vector database.
  - `coding`: A specialized Python code generation agent that works in a loop (Write -> Execute -> Verify).

### 2. Retrieval-Augmented Generation (RAG) Pipeline
- **Vector Database**: Qdrant is used to store document embeddings.
- **Document Registry**: `document_registry.db` tracks document IDs, paths, and metadata to map chunks back to their original source files.
- **Ingestion**: Documents are chunked, embedded, and stored in Qdrant with links back to the original files in the registry.

### 3. The Coding Sandbox Pipeline
- **Code Generation**: The LLM writes raw Python code to solve data analysis, math, or document generation tasks.
- **Docker Sandbox Execution**: The generated code is executed inside a temporary, isolated, read-only Docker container (`coding-agent-sandbox`) with no internet access and strict resource limits.
- **Verification Loop**: If the code fails (syntax error, runtime error), the stderr is captured and fed back to the LLM for self-correction (up to 3 attempts).

## ✨ Features

### Active Features
- **Local LLM Inference**: Fully powered by local Ollama models (e.g., `qwen2.5-coder:3b`, `llama3.1`).
- **Dynamic File Uploads**: Users can attach files (PDF, CSV, Excel, Word, PPT) directly in the chat UI. These files are mounted into the Coding Sandbox for immediate, full-document data analysis (bypassing vector search for exact processing).
- **Document Generation (Tool Layer)**: The Coding Agent can generate new files (Presentations, Spreadsheets, Word Docs) inside its sandbox. These are automatically extracted to a persistent host directory and served as download links in the chat stream.
- **Dynamic Model Selection**: Users can choose between "Auto (Supervisor)" routing or explicitly force a specific open-source model (e.g., `qwen3`) via the chat UI. The Supervisor will respect this override and instruct the chosen worker agent to use the selected model.
- **Session History**: Fully functional chat history persistence using browser `localStorage`. Users can create new sessions, browse past chats in the sidebar, and resume old conversations.
- **Deepseek-Style UI**: An ultra-sleek, dark-mode React frontend with a full-screen layout, sliding sidebars, floating pill input fields, and trajectory visualization.
- **Execution Trajectory**: A dedicated UI tab that visualizes the agentic flow, showing timestamped logs of how the Supervisor routed the task and which sandbox containers were spun up.
- **Session Logs**: Users can download a `.zip` archive of their backend server logs directly from the UI header for debugging and audit purposes.

---
*This file will be continually updated as new features and architectural changes are merged.*

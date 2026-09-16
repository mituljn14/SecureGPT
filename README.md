# SecureGPT— Sovereign On-Premise Agentic AI Workbench

Enterprise RAG over a private corpus (Nova Automobile Engineering Solutions),
running entirely on one machine: hybrid dense+sparse retrieval in Qdrant with
role-based access enforced inside the vector filter, a LangGraph supervisor
that routes to four agents (enterprise RAG, coding in a network-less Docker
sandbox, vision, meeting minutes), local LLMs through Ollama, per-step audit
trail, and a ChatGPT-style React UI with real username/password login.

Built for SIH 2026 (PS SIH26117) by Team ISOLATE.

> **Read this first.** A `git clone` gives you the code but **not the data**
> (`corpus/`, `markdown/`, `chunks/`, `embeddings/` are git-ignored). You need
> the [data package on Google Drive](https://drive.google.com/drive/folders/1dGr-Q_HlQI098W--t3xZdUQFslbiU42K) *and* a machine with Docker,
> Ollama, Python 3.13 and Node. Follow the steps in order; each one is checked
> by a command you can run.

---

## Contents

1. [What you get](#1-what-you-get)
2. [Requirements](#2-requirements)
3. [Setup, step by step](#3-setup-step-by-step)
4. [Running it](#4-running-it)
5. [Using the app](#5-using-the-app)
6. [Configuration reference](#6-configuration-reference)
7. [Project layout](#7-project-layout)
8. [How a query flows](#8-how-a-query-flows)
9. [Evaluation (RAGAS)](#9-evaluation-ragas)
10. [Troubleshooting](#10-troubleshooting)
11. [Things to keep in mind](#11-things-to-keep-in-mind)

---

## 1. What you get

| Piece | Where | What it does |
|---|---|---|
| **API** | `backend/retrieval_pipeline/api` | FastAPI on `127.0.0.1:8001` — login, chat, history, corpus search, stats, audit |
| **Supervisor + agents** | `backend/retrieval_pipeline/graph` | LangGraph: routes each query to enterprise / coding / vision / meeting, or answers directly |
| **Retrieval** | `backend/retrieval_pipeline/retrieval` | BGE-M3 hybrid search in Qdrant, RBAC filter, exact-identifier lookup, CRAG + multi-agent escalation |
| **Coding agent** | `backend/retrieval_pipeline/graph/agents/coding_agent` | Finds the file, writes pandas code, runs it in Docker with `--network none`, verifies the result, exports files |
| **Vision agent** | `backend/vision_agent` | Images, PDFs, PPTX/DOCX, spreadsheets → OCR/extraction, template charts, answers |
| **Meeting agent** | `backend/meeting_agent` | Audio → Whisper transcript → diarization → role-scoped RAG → summary/decisions/actions → DOCX |
| **Frontend** | `frontend/` | React + Vite, ChatGPT-style shell, login, Trace, Security audit, Documents, Admin |
| **Ingestion** | `pipeline/`, `scripts/`, `config/` | Converters → chunking → embedding → Qdrant upsert (only needed to rebuild the corpus) |
| **Evaluation** | `backend/evaluation` | RAGAS scoring of the enterprise agent, separate from the runtime pipeline |
| **Docs** | `docs/query-lifecycle` | Diagrams of the full request path (PDF + PNG) |

---

## 2. Requirements

### Hardware
- 16 GB RAM minimum (Qdrant holds the 763k-point index; BGE-M3 + a 3B LLM run on CPU). 32 GB is comfortable.
- ~15 GB free disk: data package 4.8 GB, Qdrant index ~5 GB, models ~6 GB, Python env ~5 GB.
- No GPU needed. Everything below was developed and demoed on a CPU-only Windows 11 laptop.

### Software

| Tool | Version used | Notes |
|---|---|---|
| **Python** | 3.13.7 | `requirements-py313.txt` is written for 3.13. 3.11/3.12 also work with the same file. |
| **Node.js / npm** | 24.x / 11.x | 18+ is fine. |
| **Docker Desktop** | current | Runs Qdrant **and** the coding-agent sandbox. Must be running before the backend starts. |
| **Ollama** | current | Serves every LLM at `http://localhost:11434`. |
| **Git** | any | |

**Not** required: Tesseract (RapidOCR is pure pip; Tesseract is only an optional fallback), ffmpeg (bundled through `imageio-ffmpeg`), any cloud API key.

### The data package (download from Google Drive)

**https://drive.google.com/drive/folders/1dGr-Q_HlQI098W--t3xZdUQFslbiU42K**

Download the whole `SecureGPT-v2-data-package` folder (≈ 9.3 GB). It contains its own `README-DATA.md` and `checksums.sha256`:

| File | Size | Unzips to | Needed for |
|---|---|---|---|
| `corpus.zip` | 0.14 GB | `<repo>/corpus/` — 6,641 documents | Coding agent (reads the real file), vision agent corpus lookups |
| `markdown.zip` | 0.04 GB | `<repo>/markdown/` — 6,641 `.md` mirrors | Coding agent (the exact text that was indexed) |
| `chunks.zip` | 0.19 GB | `<repo>/chunks/` — `chunks.jsonl`, 763,497 lines | Rebuilding the Qdrant collection (option A) |
| `embeddings.zip` | 1.64 GB | `<repo>/embeddings/` — 20 `.npz` shards, dense + sparse | Rebuilding the Qdrant collection (option A) |
| `qdrant-snapshot/securegpt_v2-4614275046930232-2026-09-16-09-52-30.snapshot` | 7.33 GB | restored straight into Qdrant | Ready-made index (option B, fastest) |

`corpus.zip` and `markdown.zip` are always required. For the vector index take **either** the snapshot (step 3.4b, ~10 min) **or** `chunks.zip` + `embeddings.zip` (step 3.4a, 20–40 min). Verify what you received first:

```bash
cd SecureGPT-v2-data-package
sha256sum -c checksums.sha256      # Git Bash / macOS / Linux; PowerShell: Get-FileHash <file>
```

---

## 3. Setup, step by step

All commands are shown for **Windows PowerShell** unless noted; the macOS/Linux
equivalents are the usual ones (`source venv/bin/activate`, forward slashes).

### 3.1 Clone and place the data

```bash
git clone https://github.com/Harsh-Go-yal/secure-gpt-v2.git
cd secure-gpt-v2
# each zip contains its top-level folder: extract them INTO the repo root
tar -xf ../SecureGPT-v2-data-package/corpus.zip
tar -xf ../SecureGPT-v2-data-package/markdown.zip
# only if you are going to rebuild the index (step 3.4a) instead of restoring the snapshot:
tar -xf ../SecureGPT-v2-data-package/chunks.zip
tar -xf ../SecureGPT-v2-data-package/embeddings.zip
```
(`tar` ships with Windows 10/11, macOS and Linux; any zip tool works.)

Check:
```bash
dir corpus markdown
```
You should see `corpus/Executive_Corporate`, `corpus/Finance`, … and `markdown/` with the same tree and `.md` appended to every file name (e.g. `corpus/Operations/COO/Production_Reports/Work_Order_Data_Q3_2022.csv` ↔ `markdown/…/Work_Order_Data_Q3_2022.csv.md`).

### 3.2 Ollama — pull the models

Install Ollama, make sure it is running (`ollama serve` if it isn't a service), then:

```bash
ollama pull llama3.2:3b        # supervisor, enterprise agent, general answers
ollama pull qwen2.5-coder:3b   # coding agent
ollama pull moondream          # vision agent (image description)
ollama pull ministral-3:8b     # optional: local RAGAS judge for evaluation
```

Check:
```bash
curl http://localhost:11434/api/tags
```

The **embedding model is not from Ollama**: `BAAI/bge-m3` (~2.3 GB) is downloaded from Hugging Face by `FlagEmbedding` the first time the backend embeds a query. That first query therefore takes a few minutes; every later start reads it from the HF cache. If the machine has no internet at all, copy `~/.cache/huggingface/hub/models--BAAI--bge-m3` from a machine that has it.

### 3.3 Qdrant (vector database)

```bash
docker run -d --name securegpt-qdrant -p 6333:6333 -p 6334:6334 \
  -v securegpt_qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

Check: `curl http://localhost:6333/collections` → `{"result":{"collections":[]}}` (empty until step 3.4).

The named volume `securegpt_qdrant_data` keeps the index across container restarts; after a reboot just `docker start securegpt-qdrant`.

### 3.4 Python backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-py313.txt
```

- `requirements-py313.txt` is the curated list (Python 3.11–3.13).
- `requirements-lock.txt` is the exact `pip freeze` of the environment this was demoed in (Python 3.13, Windows, CPU torch). Use it if the curated file resolves to something that misbehaves: `pip install -r requirements-lock.txt`.
- `requirements.txt` is the historical file with torch 2.1 pins; it does **not** install on Python 3.13. Leave it.

Then create your environment file:

```powershell
copy .env.example .env
```
Open `backend/.env` and set `JWT_SECRET_KEY` to any long random string. Every other value has a working default for a local setup (see §6).

Check the environment imports cleanly (takes ~60–90 s the first time because of torch):
```powershell
python -c "import retrieval_pipeline.graph.workflow as w; print('graph ok')"
```

### 3.4a Load the corpus into Qdrant — option A, rebuild from `chunks/` + `embeddings/`

Skip this if you restore the snapshot in 3.4b (faster). From the **repo root**, with the venv active:

```powershell
python pipeline/upsert_shards.py
```

It streams `chunks/chunks.jsonl` once, joins each 40,000-row shard to its `.npz` vectors and upserts them. It is resumable (`embeddings/_upserted.json` records finished shards) and idempotent (chunk ids are deterministic). Expect 20–40 minutes on a laptop; peak RAM ~300 MB plus Qdrant's own ~4 GB.

Check:
```bash
curl http://localhost:6333/collections/securegpt_v2
```
`points_count` must be **763497** and `status` `green`.

### 3.4b Load the corpus into Qdrant — option B, restore the snapshot (recommended)

The package's `qdrant-snapshot/` folder holds the demo collection exactly as it was used (`securegpt_v2-4614275046930232-2026-09-16-09-52-30.snapshot`, 7.33 GB). Copy it into the running container and ask Qdrant to recover it:

```bash
cd SecureGPT-v2-data-package
docker cp qdrant-snapshot/securegpt_v2-4614275046930232-2026-09-16-09-52-30.snapshot securegpt-qdrant:/qdrant/snapshots/

curl -X PUT "http://localhost:6333/collections/securegpt_v2/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location":"file:///qdrant/snapshots/securegpt_v2-4614275046930232-2026-09-16-09-52-30.snapshot"}'
```
The recover call runs for a few minutes and returns `{"result":true,...}`. Same check as above: `points_count` **763497**, status `green`. Nothing from `chunks/` or `embeddings/` is needed for this path.

(To *create* a fresh snapshot on a machine that has the collection: `curl -X POST "http://localhost:6333/collections/securegpt_v2/snapshots?wait=true"`, then `docker cp securegpt-qdrant:/qdrant/snapshots/securegpt_v2/<name> .`)

### 3.5 Coding-agent sandbox image

The coding agent runs generated Python inside a throw-away container with no network. Build the image once:

```powershell
cd backend\retrieval_pipeline\graph\agents\coding_agent
docker build -t coding-agent-sandbox -f Dockerfile .
```

Use `Dockerfile` (python 3.11-slim + pandas/numpy/openpyxl/pymupdf/python-docx, runs as a non-root user with `/workspace` as the working directory — the paths the generated code expects). `Dockerfile.sandbox` is an older variant; don't use it.

Check: `docker images coding-agent-sandbox` shows the image (~600 MB). Without it, coding queries fail with "image not found".

### 3.6 Frontend

```powershell
cd frontend
npm install
```

`frontend/src/config.js` holds the backend URL (`http://127.0.0.1:8001`). Change it only if you run the API elsewhere.

---

## 4. Running it

Three things must already be up: **Docker Desktop** (Qdrant container started), **Ollama**, and the data folders in place. Then, in two terminals:

**Backend**
```powershell
cd backend
venv\Scripts\activate
python -u -m uvicorn retrieval_pipeline.api.main:app --host 127.0.0.1 --port 8001
```
Cold start takes **60–120 s** (torch + FlagEmbedding import) before `Uvicorn running on http://127.0.0.1:8001` appears. That is normal.

**Frontend**
```powershell
cd frontend
npm run dev
```
Open **http://localhost:5173** — use `localhost`, not `127.0.0.1` (Vite binds IPv6 by default on Windows).

**Health check in one line**
```bash
curl http://127.0.0.1:8001/roles
```
Every role must show a non-zero `chunks` count (CEO 763,497 … CISO 1,110). **If every role shows 0, Qdrant isn't reachable** — start Docker Desktop / the container and restart the backend.

After a reboot the order is: Docker Desktop → `docker start securegpt-qdrant` → Ollama → backend → frontend.

---

## 5. Using the app

### Accounts

Sixteen demo accounts, password **`pass123`** for all of them. Access is enforced on the server from the role in the signed token, not by the UI.

| Department | Username | Role | Readable chunks |
|---|---|---|---|
| Executive Corporate | `ceo_user` | CEO — reads everything | 763,497 |
| | `rmd_user` | Regional Managing Directors | 297,124 |
| | `coo_user` | COO (Operations documents) | 120,324 |
| | `corp_strat` | Corporate Strategy | 9,226 |
| | `pmo_user` | Enterprise PMO | 5,346 |
| | `corp_comm` | Corporate Communications | 1,222 |
| Technology & Digital | `cdo_user` | CDO | 199,973 |
| | `cto_user` | CTO | 60,614 |
| | `cengo_user` | CEngO | 27,250 |
| | `cio_user` | CIO | 17,581 |
| | `ciso_user` | CISO | 1,110 |
| Finance | `cfo_user` | CFO | 1,152 |
| Legal & Compliance | `clo_user` | CLO | 16,702 |
| People & Culture | `chro_user` | CHRO | 2,811 |
| Sales & Commercial | `cco_user` | CCO | 1,716 |
| | `cmo_user` | CMO | 1,346 |

Counts are live on the **Admin** page. Users are defined in `backend/retrieval_pipeline/api/database.py`.

### Queries that are verified to work

| As | Prompt | Expect |
|---|---|---|
| `coo_user` | `Which plant and line handled work order WO-202207-000342?` | Chennai Plant, Trim & Final Assembly Line 2 — via the exact-identifier lookup, ~10 s |
| `coo_user` | `What was the scrap quantity for work orders in Q3 2022?` | Lists rows 550/342/635 with scrap 13/22/15 |
| `chro_user` | same question | Honest "not found" — HR cannot read Operations data (RBAC demo) |
| `coo_user` | `From Work_Order_Data_Q3_2022.csv, calculate the total scrap quantity, the total actual quantity, and the scrap rate as a percentage of total actual quantity. Also report which plant has the highest total scrap quantity and how much.` | Coding agent: 10,013 · 360,931 · 2.77 % · Pune Plant 2 (1,966), metric cards, CSV export, **View code** |
| any | attach a `.png`/`.pdf` and ask about it | Vision agent |
| any | attach a `.csv` and ask for a bar chart | Vision agent draws it inline |
| any | attach an `.mp3`/`.wav` recording | Meeting agent → minutes DOCX |

Typical latency on CPU: 10–60 s for fast-path answers, 1–3 min when CRAG/multi-agent escalation or the coding sandbox is involved.

### Reading the UI
- **Chat | Trace** (top centre): Trace expands every step under every answer — node, decision, thresholds, documents and chunk ids used, network target.
- **via fast path / CRAG / multi-agent** pill: which retrieval stage's chunks reached the prompt.
- **Local / External** (top right): computed from real trajectory events, never asserted.
- **Security** page: the signed-in user's last 100 audit events across all their chats, plus the runtime targets.
- **Documents**: raw corpus search scoped to the signed-in role.

---

## 6. Configuration reference

All settings are environment variables read from `backend/.env` (template: `backend/.env.example`). Nothing is hardcoded in the UI — roles, models, file types and counts all come from the API.

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://localhost:6333` / `securegpt_v2` | Vector store |
| `LLM_DEPLOYMENT_TYPE` | `on_premise` | `on_premise` uses the Ollama endpoint below; `cloud` uses litellm with `CLOUD_MODEL` and is flagged **external** in every trace |
| `ON_PREMISE_API_BASE` / `ON_PREMISE_MODEL` | `http://localhost:11434/v1` / `openai/llama3.2:3b` | Supervisor, enterprise and general agents (OpenAI-compatible Ollama endpoint; keep the `openai/` prefix) |
| `OLLAMA_MODEL` | `qwen2.5-coder:3b` | Coding agent |
| `VISION_MODEL` / `TEXT_MODEL` | `moondream` / `llama3.2:3b` | Vision agent |
| `WHISPER_MODEL` | `base` | Meeting agent STT (`small`/`medium` are slower but better) |
| `JWT_SECRET_KEY` | — | **Set it.** Signs login tokens (60-minute expiry) |
| `PROTOTYPE_AUTH` | `1` | Enables `POST /login/demo` (one-click role tokens). Set `0` outside demos |
| `RBAC_SCOPE` | `role` | `role`: only chunks tagged with the user's role (+ CEO everything). `department`: also every chunk of the user's department |
| `COMPANY_NAME` / `COMPANY_ALIASES` | Nova … | Used by the supervisor prompt and the "our / we / the company" routing rule |
| `DENSE_RELEVANCE_THRESHOLD` | `0.50` | Floor for "the corpus has a real match" |
| `FAST_PATH_CONFIDENCE_THRESHOLD` | `0.55` | Fast path accepted without grading above this; below it CRAG runs |
| `VISION_SOURCE_THRESHOLD` | `0.65` | Bar for the vision agent to open a corpus document on its own |
| `RETRIEVAL_TOP_K` / `MAX_CONTEXT_CHUNKS` | `5` / `5` | Chunks retrieved / forwarded to the prompt |
| `IDENTIFIER_LOOKUP_LIMIT` / `IDENTIFIER_MIN_LEN` | `3` / `5` | Exact record-code lookup |
| `CORPUS_ROOT` / `MARKDOWN_ROOT` | `<repo>/corpus`, `<repo>/markdown` | Only if the data lives elsewhere |
| `RAGAS_JUDGE_*`, `RUNPOD_API_KEY` | — | Evaluation only |

Changing any of these requires a backend restart.

---

## 7. Project layout

```
secure-gpt-v2/
├── backend/
│   ├── .env.example              ← copy to .env
│   ├── requirements-py313.txt    ← install this
│   ├── requirements-lock.txt     ← exact versions of the demo environment
│   ├── retrieval_pipeline/
│   │   ├── api/                  FastAPI: routes/, chat_store.py (SQLite history), security.py
│   │   ├── graph/                LangGraph: workflow.py, nodes.py, agents/, telemetry.py
│   │   │   └── agents/coding_agent/   agent_graph.py, sandbox_executor.py, Dockerfile.sandbox
│   │   ├── retrieval/            retrieve.py, identifier_lookup.py, filter_builder.py, config.py
│   │   ├── context/              prompt + context builder
│   │   └── llm/provider.py       Ollama / cloud switch
│   ├── vision_agent/             standalone vision pipeline, wired to v2 retrieval
│   ├── meeting_agent/            Whisper STT → minutes
│   ├── evaluation/               RAGAS eval (dataset.py, ragas_eval.py, runpod_judge.py, results/)
│   ├── data/chat_history.db      created on first run (git-ignored)
│   └── logs/session.jsonl        full audit log (api_request, node_enter/exit, trajectory, llm calls)
├── frontend/src/                 Workbench.jsx, LoginPage.jsx, SecurityPage.jsx, DocumentsPage.jsx,
│                                 AdminPage.jsx, Markdown.jsx, icons.jsx, api.js, config.js, theme.css
├── pipeline/                     ingestion (convert → chunk → embed → upsert_shards.py)
├── config/departments.yaml       department / role mapping used at ingest
├── docs/query-lifecycle/         architecture diagrams (PDF, PNG)
├── corpus/  markdown/  chunks/  embeddings/    ← data package, NOT in git
└── FEATURES.md                   feature list
```

---

## 8. How a query flows

See `docs/query-lifecycle/SecureGPT-Query-Lifecycle.pdf` for the diagrams. In one paragraph:

`POST /chat` → JWT gives username/role/department → **supervisor** (greeting? attachment type? one LLM classification; JSON parsed in any shape; "our/we/Nova" forces enterprise; corpus probe overrides a wrong "general") → for enterprise: **fast_retrieval** (hybrid top-5, confidence = best dense cosine, exact-ID hits first) → ≥ 0.55 accept, else **CRAG** (15-chunk pool, 3 windows graded by the LLM) → ≥ 0.7 accept, else **multi-agent** (3 rewrites, sparse rerank, per-chunk verification) → **build_context** → **enterprise agent** answers only from context. Coding / vision / meeting go straight to their agent. Every node writes `node_enter`/`node_exit` to `logs/session.jsonl`, every step carries its node, network target and local/external flag, and the turn is stored per user in SQLite.

---

## 9. Evaluation (RAGAS)

Lives in `backend/evaluation/`, deliberately separate from the runtime pipeline.

```powershell
cd backend
venv\Scripts\activate
python -m evaluation.ragas_eval --only enterprise      # 10 enterprise rows
python -m evaluation.ragas_eval --only routing         # routing-only rows, no LLM judge
```

The judge defaults to local `ministral-3:8b`. A 3B judge echoes the schema and scores nothing; for meaningful numbers use an 8B+ judge, or point `RAGAS_JUDGE_API_BASE/MODEL/API_KEY` at any OpenAI-compatible endpoint (a `Qwen2.5-14B-Instruct-AWQ` on a rented GPU was used for the committed run in `evaluation/results/`). `runpod_judge.py` can start/stop such a pod if `RUNPOD_API_KEY` is set — **terminate the pod when done**. Results land in `evaluation/results/eval_*.json` with per-row answers, retrieved-context snippets and metrics.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Login page says "Backend starting…" for a minute | Cold import of torch/FlagEmbedding | Wait; it comes alive on its own |
| `/roles` shows `chunks: 0` for everyone; answers say nothing was found | Qdrant not reachable | Start Docker Desktop, `docker start securegpt-qdrant`, restart backend |
| `points_count` is not 763,497 | Upsert incomplete | Re-run `python pipeline/upsert_shards.py` (it resumes) |
| First query hangs for minutes | BGE-M3 downloading from Hugging Face | One-time; watch the backend console |
| Coding query: "image not found" / Docker error | Sandbox image missing or Docker not running | Step 3.5; start Docker Desktop |
| Coding query fails after 3 attempts with a pandas error | The 3B coder wrote a pandas-1 idiom (e.g. multi-column `groupby[...]`) | Ask for one grouped column at a time; sums/means/max are reliable |
| Meeting agent: "not available on this machine" | `openai-whisper` not installed | `pip install openai-whisper imageio-ffmpeg` |
| `UnicodeEncodeError` in a Windows console | cp1252 console | Run with `set PYTHONIOENCODING=utf-8` (the API sets this itself) |
| Port 8001 "only one usage of each socket" | Previous backend still running | Find it with `netstat -ano \| findstr :8001`, `taskkill /PID <pid> /F` |
| Frontend loads but `127.0.0.1:5173` doesn't | Vite bound IPv6 | Use `http://localhost:5173` |
| Everything answers "External detected" | `LLM_DEPLOYMENT_TYPE=cloud` | Set `on_premise` for a fully local run |
| Token expired mid-demo (sent back to login) | 60-minute JWT | Sign in again; history is kept per user |

Where to look: the backend console prints the supervisor decision and each agent's steps; `backend/logs/session.jsonl` has everything (API calls, node timings, LLM prompts/responses).

---

## 11. Things to keep in mind

- **Order of start-up matters**: Docker Desktop → Qdrant container → Ollama → backend → frontend. The backend does not retry Qdrant; if it started before Qdrant, restart it.
- **Ports**: API on **8001** (8000 is deliberately avoided — another Docker service on the original machine publishes it), Qdrant 6333, Ollama 11434, Vite 5173.
- **Latency is CPU-bound.** A 3B model on CPU does ~10–20 tokens/s; CRAG grading is one LLM call per window. Don't run two heavy queries at once during a demo, and don't run evaluation while demoing.
- **Nothing leaves the machine** in the default configuration: Ollama, Qdrant, the sandbox (`--network none`) and Whisper are all local. The only outbound call is the one-time BGE-M3 download. The Trace shows a red "external" badge on any step that would break that.
- **Data is scoped at ingest.** Each chunk carries `access_roles` (owner role + CEO) and `department`. `RBAC_SCOPE=role` enforces exactly those tags; there is no per-query bypass.
- **History and audit are per user** and live in `backend/data/chat_history.db` (git-ignored). Deleting a chat from the sidebar removes its steps from the Security page; `logs/session.jsonl` is the permanent record.
- **`.env` is git-ignored.** Never commit it; commit changes to `.env.example` instead.
- **Runtime files that show up as modified** — `backend/logs/session.jsonl` and `coding_agent/outputs/*` — are tracked from early history but change on every run. Don't include them in commits.
- **Rebuilding the corpus from scratch** (new documents) is the `pipeline/` job: `convert_all.py` → `chunk_all.py` → embedding (`embed_pod.py` / `runpod_embed.py`, GPU recommended) → `upsert_shards.py`. `config/departments.yaml` decides the `access_roles` tag for each folder. This is a multi-hour process and not needed to run the demo.

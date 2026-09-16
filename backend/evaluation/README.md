# RAG/coding-agent correctness eval (RAGAS)

Runs the golden query set in [`dataset.py`](dataset.py) through the live
pipeline **in-process** (imports `retrieval_pipeline.graph.workflow.graph`
directly — no HTTP, no need for uvicorn to be running) and scores each
answer.

## Run it

```bash
cd backend
venv/Scripts/python.exe -m evaluation.ragas_eval
```

Needs Qdrant and the configured LLM backend (Ollama by default) reachable
— same dependencies the live app has. **Don't run this while the live app
has long requests in flight** (e.g. a coding-agent query mid-flight from
the frontend) — this queues up behind them on the same local LLM, same as
any other request would.

Results print to the console and are saved to `results/eval_<epoch>.json`
for diffing between runs (e.g. before/after a pipeline change).

## What it measures, and why two different checks

- **Routing correctness** (every row): did `supervisor_node` send the
  query where `dataset.py` says it should go. This alone catches the two
  biggest bugs found and fixed in this pipeline: the fast-retrieval
  confidence gate that was hardcoded to `1.0` whenever *any* hit came
  back (so the CRAG/multi-hop escalation path never actually fired), and
  the supervisor answering a real company question directly — with zero
  retrieval — because a single classification call misjudged it as
  general knowledge.
- **Enterprise/general RAG rows** — RAGAS `Faithfulness` +
  `AnswerRelevancy`, both reference-free (no hand-maintained "ground
  truth" answer needed). These directly test whether the generated
  answer is actually grounded in what was retrieved, which is the core
  failure mode this whole pipeline was rebuilt to prevent.
- **Coding-agent rows** — deterministic checks instead of RAGAS: did the
  expected numeric value show up in the answer, did a requested file
  export actually produce a download. RAGAS' context-based metrics don't
  apply here — the coding agent computes a number in a sandbox and
  verifies it against a formal result marker (see
  `coding_agent/verifier.py`), there's no free-text "answer synthesized
  from retrieved prose" to score for faithfulness.
- **`expect_no_fabrication` rows** — a query that should get an honest
  "I don't have this" rather than a confident invented answer (an
  out-of-corpus question, or a role asking about data its RBAC scope
  shouldn't surface). Checked with a phrase heuristic
  (`_looks_like_honest_decline`) — this one is inherently semantic and
  worth eyeballing the actual answer text in the JSON output, not just
  trusting the boolean.

## Extending the dataset

Add rows to `DATASET` in `dataset.py` — each is independent, see the
field docs in that file's module docstring. Growing this set is exactly
how "does this handle the next query nobody thought of yet" gets checked
automatically instead of by hand.

## Setup note (ragas + langchain_community version mismatch)

`ragas>=0.4` imports `langchain_community.chat_models.vertexai.ChatVertexAI`
unconditionally at import time. Current `langchain_community` dropped that
submodule (Vertex AI support moved to the separate
`langchain-google-vertexai` package), so a bare `import ragas` fails.
`ragas_eval.py` detects and fixes this itself on first run (writes a tiny
stub module into site-packages) — you don't need to do anything, this is
just here so the failure mode is understood if you ever see it elsewhere
(e.g. a different script that imports `ragas` before `ragas_eval.py` has
had a chance to run its shim setup).

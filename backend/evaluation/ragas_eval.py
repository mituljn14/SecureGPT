"""
ragas_eval.py

Automated correctness evaluation for the full SecureGPT pipeline
(supervisor routing -> hybrid RAG / CRAG / multi-hop -> enterprise agent,
or -> coding agent), using the RAGAS framework plus a few
SecureGPT-specific checks RAGAS itself doesn't cover.

WHY TWO DIFFERENT SCORING PATHS
--------------------------------
RAGAS' metrics (Faithfulness, AnswerRelevancy, ...) are built for a
"retrieve text, synthesize an LLM answer from it" flow -- which is exactly
the enterprise/general RAG path in this app. The coding agent is a
different shape: it computes a number deterministically in a sandbox and
verifies it against a formal ___RESULT___ marker (see coding_agent/
verifier.py) -- there's no free-text "answer synthesized from context" to
score for faithfulness against. Forcing RAGAS metrics onto that path would
just be scoring something they weren't designed for. So:

  - expect_destination in {"enterprise", "general", "__direct__"} with
    retrieved context available -> RAGAS Faithfulness + AnswerRelevancy
    (both reference-free: no hand-maintained "ground truth" needed, and
    both directly test the exact failure mode this whole session was
    about -- an answer not actually grounded in what was retrieved).
  - expect_destination == "coding" -> deterministic checks: did
    verification pass, does the answer contain the expected value(s), was
    a file produced if one was requested.
  - Every row, regardless of path -> was ROUTING correct. This alone
    catches the two biggest bugs fixed this session: the RRF-confidence-
    always-1.0 bug (fast path never escalated to agentic retrieval) and
    the supervisor's direct-answer hallucination-bypass (a real company
    question answered with zero grounding).

USAGE
-----
    cd backend
    venv/Scripts/python.exe -m evaluation.ragas_eval

Needs Qdrant + Ollama (or whatever LLM_DEPLOYMENT_TYPE points at)
reachable, same as the live app -- it invokes the LangGraph workflow
in-process (no HTTP, no running uvicorn required), so a stuck/overloaded
API server doesn't block running this, but a queued-up local LLM
absolutely still will: don't run this while the live app has long
requests in flight, or this just queues up behind them too.

Results are printed as a summary table and written to
eval/results/<timestamp>.json for later diffing between runs.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional


# ============================================================
# Force UTF-8 on stdout/stderr before anything runs the pipeline.
#
# The graph prints status lines containing non-cp1252 characters (the
# supervisor's routing markers, and rupee signs from this corpus). On a
# Windows console defaulting to cp1252 a single such print raises
# UnicodeEncodeError, which surfaces here as "Pipeline raised: 'charmap'
# codec can't encode character" -- every row failing for a reason that has
# nothing to do with retrieval quality. api/main.py does the same thing for
# the server; an eval that imports the graph directly needs it too.
# ============================================================
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # already wrapped, or not reconfigurable (e.g. under pytest)


# ============================================================
# ragas>=0.4 compatibility shim -- MUST run before `import ragas`.
#
# ragas/llms/base.py unconditionally does
#   from langchain_community.chat_models.vertexai import ChatVertexAI
# but current langchain_community dropped that submodule (Vertex AI moved
# to the separate langchain-google-vertexai package). Without this, `import
# ragas` fails outright, even though this project never touches Vertex AI
# -- it talks to Ollama via langchain_openai.ChatOpenAI. Self-healing so a
# fresh `pip install -r requirements.txt` + run doesn't need manual
# venv surgery.
# ============================================================
def _ensure_ragas_vertexai_shim() -> None:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import langchain_community.chat_models as _cm

    shim_path = Path(_cm.__file__).parent / "vertexai.py"
    if shim_path.exists():
        return
    shim_path.write_text(
        '"""Compatibility shim -- see evaluation/ragas_eval.py '
        '(_ensure_ragas_vertexai_shim). Auto-generated."""\n\n\n'
        "class ChatVertexAI:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise ImportError(\n"
        '            "ChatVertexAI is not available -- this is an auto-generated "\n'
        '            "compatibility shim. Install the real langchain-google-vertexai "\n'
        '            "package if Vertex AI support is actually needed."\n'
        "        )\n",
        encoding="utf-8",
    )
    print(f"[ragas_eval] Created missing langchain_community vertexai shim at {shim_path}")


_ensure_ragas_vertexai_shim()

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.llms.base import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
)

# ============================================================
# Project imports
# ============================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/

from retrieval_pipeline.graph.workflow import graph  # noqa: E402
from retrieval_pipeline.retrieval.embedding import embed_query_hybrid  # noqa: E402
from evaluation.dataset import DATASET  # noqa: E402


# ============================================================
# Role -> department (api/database.py MOCK_USERS, kept in sync by hand --
# small, static, changes only when new roles are added there)
# ============================================================
ROLE_TO_DEPARTMENT = {
    "CEO": "Executive_Corporate",
    "COO": "Executive_Corporate",
    "Regional_Managing_Directors": "Executive_Corporate",
    "Corporate_Strategy": "Executive_Corporate",
    "Corporate_Communications": "Executive_Corporate",
    "Enterprise_PMO": "Executive_Corporate",
    "CTO": "Technology_Digital",
    "CIO": "Technology_Digital",
    "CISO": "Technology_Digital",
    "CDO": "Technology_Digital",
    "CEngO": "Technology_Digital",
    "CFO": "Finance",
    "CCO": "Sales_Commercial",
    "CMO": "Sales_Commercial",
    "CLO": "Legal_Compliance",
    "CHRO": "People_Culture",
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"

NOT_FOUND_PHRASES = [
    "could not find", "cannot find", "couldn't find", "don't have", "do not have",
    "no information", "not available in", "unable to find", "not found in",
    "does not appear", "no data", "don't have access", "no relevant company data",
    "outside my", "i cannot access", "i don't have enough", "not present in",
]


def _looks_like_honest_decline(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in NOT_FOUND_PHRASES)


def _chunk_text(chunk: Any) -> str:
    if hasattr(chunk, "text"):
        return chunk.text or ""
    if isinstance(chunk, dict):
        return chunk.get("text") or ""
    return ""


# ============================================================
# RAGAS LLM + embeddings, wired to the same backend the app itself uses
# ============================================================
def _build_ragas_llm():
    """LLM used to GRADE answers -- deliberately separate from the pipeline's.

    RAGAS drives structured output through instructor in `Mode.JSON`
    (ragas/llms/base.py), which pastes the Pydantic JSON Schema into the
    prompt rather than using the tool-calling API. A 3B model handed a schema
    frequently echoes the schema back instead of instantiating it, which
    surfaces as `validation error for StatementGeneratorOutput: statements
    Field required` and burns all four retries. Judging therefore wants a
    bigger model than serving does.

    The judge is configured by its own variables so a GPU endpoint can be
    used for evaluation WITHOUT touching what the pipeline runs on:

        RAGAS_JUDGE_API_BASE   OpenAI-compatible URL (vLLM/TGI on a GPU box,
                               or any hosted endpoint). Default: the local
                               Ollama the pipeline already uses.
        RAGAS_JUDGE_MODEL      Model name at that endpoint.
        RAGAS_JUDGE_API_KEY    Key, if the endpoint needs one.

    With none of them set, behaviour is exactly as before -- local Ollama,
    same model as the pipeline. Nothing changes unless you opt in.
    """
    api_base = os.getenv("RAGAS_JUDGE_API_BASE") or os.getenv(
        "ON_PREMISE_API_BASE", "http://localhost:11434/v1")
    # Default judge is the largest LOCAL model, not the pipeline's 3B. The
    # 3B echoed the JSON schema back on 3 of 5 rows (StatementGeneratorOutput
    # validation errors); an 8B clears that cliff without a GPU and without
    # any corpus text leaving the machine.
    model = (os.getenv("RAGAS_JUDGE_MODEL") or "ministral-3:8b").removeprefix("openai/")
    api_key = (os.getenv("RAGAS_JUDGE_API_KEY")
               or os.getenv("ON_PREMISE_API_KEY", "ollama"))

    remote = bool(os.getenv("RAGAS_JUDGE_API_BASE"))
    print(f"[ragas_eval] judge: {model} @ {api_base}"
          f"{'  (REMOTE -- retrieved corpus text and answers are sent there)' if remote else '  (local)'}")

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    return llm_factory(model, provider="openai", client=client)


class BGEM3RagasEmbedding(BaseRagasEmbedding):
    """Reuses this project's own BGE-M3 dense embeddings (embed_query_hybrid)
    instead of requiring a separate embedding model/endpoint -- same vector
    space the corpus itself was indexed in, and no extra service to stand up."""

    def embed_text(self, text: str, **kwargs) -> List[float]:
        dense, _sparse = embed_query_hybrid(text)
        return dense

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        return await asyncio.to_thread(self.embed_text, text)


# ============================================================
# Run one dataset row through the live pipeline (in-process, no HTTP)
# ============================================================
async def _run_pipeline(row: dict) -> dict:
    role = row["role"]
    initial_state = {
        "query": row["query"],
        "role": role,
        "department": ROLE_TO_DEPARTMENT.get(role),
        "trajectory_log": [],
        "uploaded_files": list(row.get("uploaded_files") or []),
        "generated_files": [],
        "model_override": None,
    }

    # routing_only rows test the supervisor's decision and nothing else. The
    # attachment fast-path decides from the filename before any model runs,
    # so the files need not exist -- and running the vision/meeting agents
    # on a non-existent file would just measure their error handling.
    if row.get("routing_only"):
        from retrieval_pipeline.graph.agents.supervisor import supervisor_node
        return await asyncio.to_thread(supervisor_node, initial_state)

    final_state = await asyncio.to_thread(graph.invoke, initial_state)
    return final_state


# How much of each retrieved chunk to keep in the results JSON.
CONTEXT_SNIPPET_CHARS = int(os.getenv("RAGAS_CONTEXT_SNIPPET_CHARS", "600"))


async def _evaluate_row(row: dict, llm, embeddings) -> dict:
    result = {
        "query": row["query"],
        "role": row["role"],
        "expect_destination": row["expect_destination"],
        "note": row.get("note", ""),
    }

    t0 = time.perf_counter()
    try:
        final_state = await _run_pipeline(row)
    except Exception as e:
        result["error"] = f"Pipeline raised: {e}"
        result["routing_correct"] = False
        return result
    result["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    actual_destination = final_state.get("agent_destination")
    answer = final_state.get("answer", "") or ""
    contexts = [_chunk_text(c) for c in (final_state.get("retrieved_chunks") or [])]
    contexts = [c for c in contexts if c]

    result["actual_destination"] = actual_destination
    result["routing_correct"] = (actual_destination == row["expect_destination"])
    result["answer"] = answer
    result["num_contexts"] = len(contexts)
    # Keep what the judge actually saw. A 0.0 faithfulness on an answer that
    # is verifiably correct (10-row GPU run, row 9) cannot be diagnosed from
    # a count alone -- was the fact in the chunks or not?
    result["contexts"] = [
        {"source": getattr(c, "source_path", None) or (getattr(c, "payload", {}) or {}).get("source_path"),
         "text": t[:CONTEXT_SNIPPET_CHARS]}
        for c, t in zip(final_state.get("retrieved_chunks") or [], contexts)
    ]

    if row.get("routing_only"):
        result["ragas_skipped_reason"] = "routing-only row"
        return result

    # ---- coding-agent path: deterministic checks, no RAGAS ----
    if row["expect_destination"] == "coding":
        if row.get("expect_answer_contains"):
            result["value_correct"] = any(v in answer for v in row["expect_answer_contains"])
        if row.get("expect_generated_file"):
            result["file_generated"] = bool(final_state.get("generated_files"))
        if row.get("expect_no_fabrication"):
            result["honest_decline"] = _looks_like_honest_decline(answer) or "error" in str(final_state.get("error", "")).lower()
        return result

    # ---- enterprise/general/__direct__ path ----
    if row.get("expect_no_fabrication"):
        result["honest_decline_or_grounded"] = (
            _looks_like_honest_decline(answer) if not contexts else True
        )

    if contexts and answer.strip():
        try:
            faith = await Faithfulness(llm=llm).ascore(
                user_input=row["query"], response=answer, retrieved_contexts=contexts
            )
            result["faithfulness"] = round(faith.value, 3)
        except Exception as e:
            result["faithfulness_error"] = str(e)

        try:
            # AnswerRelevancy takes no contexts in ragas 0.4 -- it judges the
            # answer against the question alone (generates candidate
            # questions from the answer and embeds them against the real one).
            relevancy = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
                user_input=row["query"], response=answer
            )
            result["answer_relevancy"] = round(relevancy.value, 3)
        except Exception as e:
            result["answer_relevancy_error"] = str(e)

        # ---- retrieval-quality metrics: need a verified reference answer ----
        # Faithfulness and relevancy only say whether the ANSWER matches the
        # CONTEXT. They cannot tell you the retriever fetched the wrong
        # documents -- an answer faithful to irrelevant chunks scores well.
        # ContextPrecision (are the retrieved chunks relevant?) and
        # ContextRecall (was the needed evidence retrieved at all?) close
        # that gap, but both compare against ground truth, so they run only
        # on rows that carry a verified one.
        gt = row.get("ground_truth")
        if gt:
            try:
                prec = await ContextPrecision(llm=llm).ascore(
                    user_input=row["query"], reference=gt, retrieved_contexts=contexts
                )
                result["context_precision"] = round(prec.value, 3)
            except Exception as e:
                result["context_precision_error"] = str(e)
            try:
                rec = await ContextRecall(llm=llm).ascore(
                    user_input=row["query"], reference=gt, retrieved_contexts=contexts
                )
                result["context_recall"] = round(rec.value, 3)
            except Exception as e:
                result["context_recall_error"] = str(e)
    else:
        result["ragas_skipped_reason"] = "no retrieved context and/or empty answer"

    return result


def _select_rows(argv):
    """Filter the golden set by expected destination.

    Scoring one agent at a time matters in practice: every row costs a full
    pipeline run plus an LLM call per RAGAS metric, all queued on the same
    local Ollama. `--only enterprise` scores just the RAG path -- which is
    also the only path RAGAS' context-based metrics actually apply to.
    """
    wanted = None
    for i, arg in enumerate(argv):
        if arg == "--only" and i + 1 < len(argv):
            wanted = argv[i + 1]
        elif arg.startswith("--only="):
            wanted = arg.split("=", 1)[1]

    if not wanted:
        return DATASET, "all"
    # "routing" selects the attachment-driven rows -- seconds to run, no LLM.
    if wanted == "routing":
        return [r for r in DATASET if r.get("routing_only")], wanted
    return [r for r in DATASET if r.get("expect_destination") == wanted], wanted


async def main():
    rows, selection = _select_rows(sys.argv[1:])
    if not rows:
        available = sorted({r.get("expect_destination") for r in DATASET})
        print(f"No rows match --only {selection!r}. Available: {available}")
        return

    llm = _build_ragas_llm()
    embeddings = BGEM3RagasEmbedding()

    print(f"Running {len(rows)} eval queries (selection: {selection}) "
          f"against the live pipeline...\n")
    results = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['query']!r} (role={row['role']})")
        r = await _evaluate_row(row, llm, embeddings)
        results.append(r)
        _print_row_result(r)
        print()

    _print_summary(results)
    _print_metric_table(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{selection}_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results written to {out_path}")


def _print_row_result(r: dict):
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return
    routing = "OK" if r["routing_correct"] else f"WRONG (expected {r['expect_destination']}, got {r['actual_destination']})"
    print(f"  routing: {routing}  ({r.get('elapsed_seconds', '?')}s)")
    if "value_correct" in r:
        print(f"  value_correct: {r['value_correct']}")
    if "file_generated" in r:
        print(f"  file_generated: {r['file_generated']}")
    if "honest_decline" in r:
        print(f"  honest_decline: {r['honest_decline']}")
    if "honest_decline_or_grounded" in r:
        print(f"  honest_decline_or_grounded: {r['honest_decline_or_grounded']}")
    if "faithfulness" in r:
        print(f"  faithfulness: {r['faithfulness']}")
    if "answer_relevancy" in r:
        print(f"  answer_relevancy: {r['answer_relevancy']}")
    if "context_precision" in r:
        print(f"  context_precision: {r['context_precision']}")
    if "context_recall" in r:
        print(f"  context_recall: {r['context_recall']}")
    for k in ("faithfulness_error", "answer_relevancy_error",
              "context_precision_error", "context_recall_error"):
        if k in r:
            print(f"  {k}: {str(r[k]).splitlines()[0][:110]}")
    if "ragas_skipped_reason" in r:
        print(f"  ragas: skipped ({r['ragas_skipped_reason']})")
    print(f"  answer: {(r.get('answer') or '')[:200]!r}")


def _mean(results, key):
    vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
    return (round(sum(vals) / len(vals), 3), len(vals)) if vals else (None, 0)


def _print_summary(results: List[dict]):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results)
    routing_ok = sum(1 for r in results if r.get("routing_correct"))
    print(f"Routing correct: {routing_ok}/{total}")

    faith_scores = [r["faithfulness"] for r in results if "faithfulness" in r]
    if faith_scores:
        print(f"Avg faithfulness ({len(faith_scores)} rows): {sum(faith_scores)/len(faith_scores):.3f}")

    rel_scores = [r["answer_relevancy"] for r in results if "answer_relevancy" in r]
    if rel_scores:
        print(f"Avg answer_relevancy ({len(rel_scores)} rows): {sum(rel_scores)/len(rel_scores):.3f}")

    value_checks = [r for r in results if "value_correct" in r]
    if value_checks:
        ok = sum(1 for r in value_checks if r["value_correct"])
        print(f"Coding-agent value correct: {ok}/{len(value_checks)}")

    file_checks = [r for r in results if "file_generated" in r]
    if file_checks:
        ok = sum(1 for r in file_checks if r["file_generated"])
        print(f"File-export produced a download: {ok}/{len(file_checks)}")

    decline_checks = [r for r in results if "honest_decline" in r or "honest_decline_or_grounded" in r]
    if decline_checks:
        ok = sum(1 for r in decline_checks if r.get("honest_decline") or r.get("honest_decline_or_grounded"))
        print(f"No-fabrication rows handled honestly: {ok}/{len(decline_checks)}")

    failures = [r for r in results if not r.get("routing_correct")
                or r.get("value_correct") is False
                or r.get("file_generated") is False
                or (r.get("honest_decline") is False)
                or (r.get("honest_decline_or_grounded") is False)]
    if failures:
        print(f"\n{len(failures)} row(s) need attention:")
        for r in failures:
            print(f"  - {r['query']!r}")




def _print_metric_table(results: List[dict]):
    """Compact per-metric roll-up. Each line shows the mean and how many
    rows actually contributed -- a mean over 2 rows is not a measurement,
    and hiding the count is how such numbers get quoted as if it were."""
    print()
    print("METRICS (mean over rows that produced a score)")
    print("-" * 52)
    for key, label in (
        ("faithfulness", "Faithfulness       (answer grounded in context)"),
        ("answer_relevancy", "Answer relevancy   (answer addresses question)"),
        ("context_precision", "Context precision  (retrieved chunks relevant)"),
        ("context_recall", "Context recall     (needed evidence retrieved)"),
    ):
        mean, n = _mean(results, key)
        shown = f"{mean:.3f}" if mean is not None else "  --"
        print(f"  {label:48} {shown}   n={n}")
    routing = [r for r in results if "routing_correct" in r]
    ok = sum(1 for r in routing if r["routing_correct"])
    print(f"  {'Routing accuracy':48} {ok}/{len(routing)}")
    decl = [r for r in results if "honest_decline_or_grounded" in r or "honest_decline" in r]
    if decl:
        good = sum(1 for r in decl if r.get("honest_decline_or_grounded") or r.get("honest_decline"))
        print(f"  {'Anti-fabrication rows handled honestly':48} {good}/{len(decl)}")
    crashed = sum(1 for r in results for k in r if k.endswith("_error"))
    if crashed:
        print(f"  {'Metric calls that errored (judge quality)':48} {crashed}")


if __name__ == "__main__":
    asyncio.run(main())

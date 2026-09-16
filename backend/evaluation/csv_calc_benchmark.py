"""
csv_calc_benchmark.py

A 10-question, ground-truthed calculation benchmark for the coding agent.

Every question targets a DIFFERENT CSV file in a DIFFERENT functional area,
and every expected value below was computed directly with pandas from the
file on disk (see `recompute_ground_truth()` -- it re-derives them at run
time, so the expectations can never silently drift from the corpus).

Each file holds 600-5,000 rows, which is the point: the answer cannot be
eyeballed from the 3-row schema preview in the prompt. The agent has to
actually mount the file and run code over all of it.

Usage:
    python -m evaluation.csv_calc_benchmark              # all 10
    python -m evaluation.csv_calc_benchmark --only 1,3,7 # subset
    python -m evaluation.csv_calc_benchmark --retrieval-only
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from retrieval_pipeline.retrieval.config import CORPUS_ROOT


# --------------------------------------------------------------------------
# The 10 cases. `op`/`column`/`where` are the ground-truth recipe -- the same
# calculation a human analyst would do -- NOT something the agent ever sees.
# --------------------------------------------------------------------------
@dataclass
class Case:
    id: int
    area: str
    rel_path: str
    question: str
    op: str
    column: str
    where: Optional[tuple] = None
    tolerance: float = 0.01          # relative tolerance on the numeric answer
    expected: Optional[float] = None  # filled by recompute_ground_truth()
    rows: Optional[int] = None


CASES = [
    Case(
        id=1,
        area="Operations / COO / Logistics",
        rel_path="Operations/COO/Logistics/Shipment_Data_Q1_2017.csv",
        question="In the Q1 2017 logistics shipment data, what is the total "
                 "shipment cost in INR across all shipments?",
        op="sum", column="Cost_INR",
    ),
    Case(
        id=2,
        area="Operations / COO / Production",
        rel_path="Operations/COO/Production_Reports/Work_Order_Data_Q1_2017.csv",
        # Names the column explicitly. Without that token, hybrid search ranks
        # the COO_*_Catalog.{csv,xlsx} index documents above the actual data
        # file -- a catalog that LISTS every production report is textually
        # similar to a question about production reports. The sparse/BM25 half
        # of the hybrid query is what separates them. See KNOWN LIMITATIONS in
        # the module docstring.
        question="What is the total Scrap_Qty across all work orders in the Q1 2017 "
                 "work order data?",
        op="sum", column="Scrap_Qty",
    ),
    Case(
        id=3,
        area="Technology / CTO / Cybersecurity",
        rel_path="Technology_Digital/CTO/Cybersecurity/"
                 "Vulnerability_Scan_Log_Battery_Management_Cloud_Service_Q1_2023.csv",
        question="In the Q1 2023 vulnerability scan log for the Battery Management "
                 "Cloud Service, what is the average CVSS score?",
        op="mean", column="CVSS_Score",
    ),
    Case(
        id=4,
        area="Technology / CTO / Machine Learning",
        rel_path="Technology_Digital/CTO/Machine_Learning/"
                 "Experiment_Log_Autonomous_Lane_Detection_Q1_2018.csv",
        question="In the Q1 2018 machine learning experiment log for autonomous lane "
                 "detection, what is the average validation accuracy of trials whose "
                 "status is Completed?",
        op="mean", column="Val_Accuracy", where=("Status", "Completed"),
    ),
    Case(
        id=5,
        area="Operations / CEngO / Electric Mobility",
        rel_path="Operations/Chief_Engineering_Officer/Electric_Mobility/"
                 "EV_Efficiency_BattPack_Q1_2017.csv",
        question="In the Q1 2017 EV battery pack efficiency test data, what is the "
                 "average efficiency percentage?",
        op="mean", column="Efficiency_Pct",
    ),
    Case(
        id=6,
        area="Operations / CEngO / Connected Vehicles",
        rel_path="Operations/Chief_Engineering_Officer/Connected_Vehicles/"
                 "CV_Events_CloudConn_Q1_2018.csv",
        question="In the Q1 2018 connected vehicle cloud connectivity event log, what "
                 "is the average latency in milliseconds?",
        op="mean", column="Latency_ms",
    ),
    Case(
        id=7,
        area="Executive / Regional MD / India",
        rel_path="Executive_Corporate/Regional_Managing_Directors/india/Nova_RMD_India/"
                 "Bengaluru/CSV_Data/Bengaluru_Regional_Operations_Registry_Part_1.csv",
        question="In the Bengaluru regional operations registry part 1, what is the "
                 "total revenue in INR across all records?",
        op="sum", column="Revenue_INR",
    ),
    Case(
        id=8,
        area="Technology / CDO / AI Models",
        rel_path="Technology_Digital/CDO/Nova_CDO_AI_Models_Subfolder_Enterprise_Package/"
                 "Nova_CDO_AI_Models/CSV_Data/Enterprise_AI_Model_Registry_Part_1.csv",
        question="In the enterprise AI model registry part 1, what is the average F1 "
                 "score across all registered models?",
        op="mean", column="F1_Score",
    ),
    Case(
        id=9,
        area="Legal / CLO / Contracts",
        rel_path="Legal_Compliance/CLO/Nova_CLO_Commercial_Contracts_Repository (1)/"
                 "Nova_CLO_Commercial_Contracts/14_Contract_Obligation_Tracker.csv",
        question="In the contract obligation tracker, what is the total contract value "
                 "across all contracts?",
        op="sum", column="Contract_Value",
    ),
    Case(
        id=10,
        area="Technology / CTO / AI Models",
        rel_path="Technology_Digital/CTO/AI_Models/"
                 "Training_Runs_Autonomous_Lane_Detection_Q2_2019.csv",
        question="In the Q2 2019 training runs for autonomous lane detection, what is "
                 "the minimum validation loss recorded?",
        op="min", column="Val_Loss",
    ),
]


def recompute_ground_truth(cases=CASES):
    """Re-derive every expected value from the CSV on disk, so this file can
    never assert a stale number if the corpus changes underneath it."""
    for c in cases:
        path = CORPUS_ROOT / c.rel_path
        if not path.exists():
            raise FileNotFoundError(f"Case {c.id}: missing corpus file {path}")
        df = pd.read_csv(path)
        sub = df if c.where is None else df[df[c.where[0]] == c.where[1]]
        if c.column not in df.columns:
            raise KeyError(f"Case {c.id}: column {c.column!r} not in {path.name}")
        c.expected = float(getattr(sub[c.column], c.op)())
        c.rows = len(df)
    return cases


# --------------------------------------------------------------------------
# Answer comparison
# --------------------------------------------------------------------------
# Scientific notation must be part of the SAME match, not left as a trailing
# "e12" -- otherwise "1.2454e+12" reads as 1.2454 and a correct answer scores
# as wrong by twelve orders of magnitude.
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")

_SCALE = {
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5,
    "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9,
    "thousand": 1e3,
}


def coerce_number(answer: Any) -> Optional[float]:
    """Pull a single number out of whatever the agent returned, honouring an
    Indian/Western magnitude word if one is attached ('878.53 crore')."""
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    if not isinstance(answer, str):
        return None

    text = answer.strip().replace("₹", " ").replace("$", " ")
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group(0).replace(",", ""))
    except ValueError:
        return None

    tail = text[m.end():].lower()
    for word, mult in _SCALE.items():
        if re.match(rf"\s*{word}\b", tail):
            return value * mult
    return value


def matches(expected: float, got: Any, tolerance: float) -> bool:
    num = coerce_number(got)
    if num is None:
        return False
    if expected == 0:
        return abs(num) < 1e-9
    return abs(num - expected) / abs(expected) <= tolerance


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
@dataclass
class Outcome:
    case: Case
    passed: bool = False
    answer: Any = None
    numeric: Optional[float] = None
    attempts: int = 0
    seconds: float = 0.0
    mounted: list = field(default_factory=list)
    right_file: bool = False
    note: str = ""


def _mounted_names(state) -> list:
    return [Path(p).name for p in (state.get("source_paths") or [])]


def run_retrieval_only(cases, role="CEO"):
    """Fast pre-flight: does retrieval even mount the right file? No LLM,
    no Docker -- seconds instead of minutes."""
    from retrieval_pipeline.graph.agents.coding_agent.agent_graph import node_retrieve

    results = []
    for c in cases:
        t0 = time.perf_counter()
        state = node_retrieve({"user_query": c.question, "role": role, "attempt": 0,
                               "trajectory_log": []})
        names = _mounted_names(state)
        target = Path(c.rel_path).name
        ok = target in names
        results.append(Outcome(case=c, passed=ok, mounted=names, right_file=ok,
                               seconds=time.perf_counter() - t0,
                               note=state.get("error") or ""))
        flag = "OK  " if ok else "MISS"
        print(f"[{flag}] Q{c.id:<2} {target}")
        if not ok:
            print(f"        mounted: {names or '(nothing)'}")
            if state.get("error"):
                print(f"        error:   {state['error']}")
    return results


def run_full(cases, role="CEO", model=None):
    """End-to-end: retrieval -> file read -> codegen -> Docker -> verify."""
    from retrieval_pipeline.graph.agents.coding_agent.agent_graph import (
        build_graph, run_agent_query,
    )

    app = build_graph()
    results = []
    for c in cases:
        target = Path(c.rel_path).name
        print("\n" + "=" * 78)
        print(f"Q{c.id}  [{c.area}]")
        print(f"  {c.question}")
        print(f"  expect {c.op}({c.column}) = {c.expected:,.4f}   over {c.rows:,} rows of {target}")
        print("=" * 78)

        t0 = time.perf_counter()
        try:
            state = run_agent_query(app, c.question, role=role, model_override=model)
        except Exception as e:
            results.append(Outcome(case=c, seconds=time.perf_counter() - t0,
                                   note=f"exception: {type(e).__name__}: {e}"))
            print(f"  -> EXCEPTION {type(e).__name__}: {e}")
            continue
        elapsed = time.perf_counter() - t0

        verification = state.get("verification") or {}
        answer = (verification.get("result") or {}).get("answer")
        if answer is None and verification.get("status") == "pass":
            # Only fall back to the prose answer when the agent actually
            # SUCCEEDED. On failure `final_answer` is a status sentence --
            # "Failed after 3 attempt(s): ..." -- and coerce_number happily
            # pulled the "3" out of it, so every failed case was reported as
            # `got=3.0`, an invented number that looked like a real (wrong)
            # result and hid what had actually gone wrong.
            answer = state.get("final_answer")

        mounted = _mounted_names(state)
        out = Outcome(
            case=c,
            answer=answer,
            numeric=coerce_number(answer),
            attempts=state.get("attempt", 0),
            seconds=elapsed,
            mounted=mounted,
            right_file=target in mounted,
            passed=matches(c.expected, answer, c.tolerance),
        )
        if not out.right_file:
            out.note = "wrong file mounted"
        elif answer is None:
            out.note = f"no answer ({verification.get('reason', 'unknown')[:60]})"
        elif not out.passed:
            out.note = "wrong value"
        results.append(out)

        print(f"  -> {'PASS' if out.passed else 'FAIL'} "
              f"got={out.numeric if out.numeric is not None else answer!r} "
              f"attempts={out.attempts} {elapsed:.1f}s")
    return results


def summarize(results, title):
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("\n" + "=" * 78)
    print(f"{title}: {passed}/{total} passed")
    print("=" * 78)
    print(f"{'Q':<3} {'area':<36} {'expected':>18} {'got':>18}  {'att':>3} {'sec':>6}  result")
    print("-" * 110)
    for r in results:
        exp = f"{r.case.expected:,.4f}" if r.case.expected is not None else "-"
        got = f"{r.numeric:,.4f}" if r.numeric is not None else (
            "-" if r.answer is None else str(r.answer)[:18])
        mark = "PASS" if r.passed else f"FAIL ({r.note})" if r.note else "FAIL"
        print(f"{r.case.id:<3} {r.case.area[:36]:<36} {exp:>18} {got:>18}  "
              f"{r.attempts:>3} {r.seconds:>6.1f}  {mark}")
    total_time = sum(r.seconds for r in results)
    print("-" * 110)
    print(f"total wall clock: {total_time:.1f}s   mean: {total_time/max(len(results),1):.1f}s/question")
    return passed, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated case ids, e.g. 1,3,7")
    ap.add_argument("--role", default="CEO")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="check file selection without running the LLM or Docker")
    ap.add_argument("--json", help="write results to this path as JSON")
    ap.add_argument("--model", help="Ollama model to generate code with, e.g. "
                                    "'ministral-3:8b'. Defaults to OLLAMA_MODEL "
                                    "(qwen2.5-coder:3b). Use to compare models on "
                                    "identical questions.")
    args = ap.parse_args()

    cases = recompute_ground_truth()
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        cases = [c for c in cases if c.id in wanted]

    print(f"Ground truth recomputed from {CORPUS_ROOT}")
    if args.model:
        print(f"Code-generation model: {args.model}")
    print()

    if args.retrieval_only:
        results = run_retrieval_only(cases, role=args.role)
        passed, total = summarize(results, "RETRIEVAL")
    else:
        results = run_full(cases, role=args.role, model=args.model)
        passed, total = summarize(results, f"END-TO-END ({args.model or 'default model'})")

    if args.json:
        Path(args.json).write_text(json.dumps([{
            "id": r.case.id, "area": r.case.area, "question": r.case.question,
            "file": Path(r.case.rel_path).name, "rows": r.case.rows,
            "expected": r.case.expected, "got": r.numeric, "raw_answer": str(r.answer),
            "passed": r.passed, "attempts": r.attempts, "seconds": round(r.seconds, 2),
            "mounted": r.mounted, "right_file": r.right_file, "note": r.note,
        } for r in results], indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

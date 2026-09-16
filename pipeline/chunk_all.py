#!/usr/bin/env python3
"""
Phase 2 driver — chunk every markdown file into chunks.jsonl, in parallel.

    .venv/Scripts/python.exe pipeline/chunk_all.py
    .venv/Scripts/python.exe pipeline/chunk_all.py --limit 50    # sample
    .venv/Scripts/python.exe pipeline/chunk_all.py --workers 8

Output: chunks/chunks.jsonl  (one JSON chunk payload per line)
        chunks/stats.json    (distribution report for review)

Pure CPU, no model loads -- minutes, not hours. The GPU is only needed
for the embedding stage that consumes this file.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.chunking.chunker import chunk_document
from pipeline.chunking.metadata import load_departments

MARKDOWN = ROOT / "markdown"
CORPUS = ROOT / "corpus"
OUT_DIR = ROOT / "chunks"
CONFIG = ROOT / "config" / "departments.yaml"

_DEPARTMENTS = None


def _init():
    global _DEPARTMENTS
    _DEPARTMENTS = load_departments(CONFIG)


def _worker(md_path_str: str):
    md_path = Path(md_path_str)
    try:
        chunks = chunk_document(
            md_path=md_path, markdown_root=MARKDOWN,
            corpus_root=CORPUS, departments=_DEPARTMENTS,
        )
        return {"ok": True, "file": str(md_path.relative_to(MARKDOWN)), "chunks": chunks}
    except Exception as e:
        import traceback
        return {"ok": False, "file": str(md_path.relative_to(MARKDOWN)),
                "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(MARKDOWN.rglob("*.md"))
    if args.limit:
        files = files[: args.limit]
    print(f"Chunking {len(files)} markdown files with {args.workers} workers...")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "chunks.jsonl"
    fail_path = OUT_DIR / "failures.jsonl"

    n_chunks = 0
    n_files = 0
    failures = 0
    by_type = Counter()
    by_dept = Counter()
    by_format = Counter()
    token_sizes = []
    tokens_by_type = defaultdict(list)
    no_roles = Counter()
    missing_fy = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as out_f, \
         open(fail_path, "w", encoding="utf-8") as fail_f, \
         multiprocessing.Pool(processes=args.workers, initializer=_init) as pool:

        for res in pool.imap_unordered(_worker, [str(p) for p in files], chunksize=8):
            if not res["ok"]:
                failures += 1
                fail_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                print(f"  ! {res['file']}: {res['error']}")
                continue

            n_files += 1
            for c in res["chunks"]:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_chunks += 1
                by_type[c["chunk_type"]] += 1
                by_dept[c["department"]] += 1
                by_format[c["source_format"]] += 1
                token_sizes.append(c["token_count"])
                tokens_by_type[c["chunk_type"]].append(c["token_count"])
                if not c["access_roles"]:
                    no_roles[c["department"]] += 1
                if c["fiscal_year"] is None:
                    missing_fy += 1

            if n_files % 500 == 0:
                el = time.time() - t0
                print(f"  [{n_files}/{len(files)}] {n_chunks:,} chunks | "
                      f"{n_files/el:.0f} files/s | ETA {(len(files)-n_files)/(n_files/el)/60:.1f} min",
                      flush=True)

    token_sizes.sort()
    p = lambda q: token_sizes[min(int(len(token_sizes) * q), len(token_sizes) - 1)] if token_sizes else 0
    stats = {
        "files_processed": n_files,
        "files_failed": failures,
        "total_chunks": n_chunks,
        "total_tokens": sum(token_sizes),
        "elapsed_sec": round(time.time() - t0, 1),
        "tokens": {"p10": p(.10), "p50": p(.50), "p90": p(.90), "p99": p(.99),
                   "max": token_sizes[-1] if token_sizes else 0,
                   "mean": round(sum(token_sizes) / max(len(token_sizes), 1), 1)},
        "by_chunk_type": dict(by_type.most_common()),
        "by_department": dict(by_dept.most_common()),
        "by_source_format": dict(by_format.most_common()),
        "tokens_by_type_mean": {k: round(sum(v) / len(v), 1) for k, v in tokens_by_type.items()},
        "chunks_missing_access_roles": dict(no_roles),
        "chunks_missing_fiscal_year": missing_fy,
        "oversize_chunks_gt_1024": sum(1 for t in token_sizes if t > 1024),
        "tiny_chunks_lt_50": sum(1 for t in token_sizes if t < 50),
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== DONE in {stats['elapsed_sec']/60:.1f} min ===")
    print(f"files: {n_files}  failed: {failures}  chunks: {n_chunks:,}  tokens: {stats['total_tokens']:,}")
    print(f"tokens p10/p50/p90/p99/max: {p(.10)}/{p(.50)}/{p(.90)}/{p(.99)}/{token_sizes[-1] if token_sizes else 0}")
    print(f"by type: {dict(by_type.most_common())}")
    print(f"oversize (>1024): {stats['oversize_chunks_gt_1024']}   tiny (<50): {stats['tiny_chunks_lt_50']}")
    print(f"missing access_roles: {sum(no_roles.values())}   missing fiscal_year: {missing_fy}")
    print(f"\nOutput: {out_path}\nStats:  {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()

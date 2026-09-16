#!/usr/bin/env python3
"""
Phase 1 driver — convert every in-scope file in corpus/ to Markdown in
markdown/, in parallel across CPU cores, writing each file's .md to disk
the moment that file finishes (not batched at the end).

Usage:
    .venv/Scripts/python.exe pipeline/convert_all.py                # full run
    .venv/Scripts/python.exe pipeline/convert_all.py --limit 20      # sample run
    .venv/Scripts/python.exe pipeline/convert_all.py --workers 8
    .venv/Scripts/python.exe pipeline/convert_all.py --force         # ignore resume cache

Output layout mirrors corpus/: corpus/A/B/report.pdf -> markdown/A/B/report.pdf.md
(the extension is preserved before ".md" so a .pdf and .xlsx with the same
base name in the same folder never collide.)

Resumable: if markdown/<...>.md already exists and is newer than the
source file, that file is skipped (safe to re-run after a crash without
redoing finished work). Use --force to reprocess everything.

IMPORTANT — thread oversubscription fix:
pymupdf4llm/PyMuPDF (and numpy's BLAS backend underneath it) spawn their own
internal thread pool per call. Left alone, each of our N worker *processes*
also runs ~3 of those internal threads, so N processes stack into ~3N
threads competing for the machine's cores -- e.g. 11 workers x ~3 threads
each on a 12-core box thrashes instead of parallelizing (observed: 12x
slower than single-process baseline, 14.6s/pdf -> ~180s/pdf). Pinning every
common thread-pool env var to 1 *before* multiprocessing.Pool spawns workers
means each worker process is strictly single-threaded internally, so the
process-level parallelism is the only parallelism -- matching the
single-threaded benchmark this file's --workers count assumes.
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.converters import pdf_to_md, docx_to_md, pptx_to_md, xlsx_to_md, csv_to_md, text_passthrough

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
MARKDOWN_OUT = ROOT / "markdown"
LOG_PATH = ROOT / "docs" / "conversion_log.jsonl"

CONVERTER_BY_EXT = {
    ".pdf": pdf_to_md.convert,
    ".docx": docx_to_md.convert,
    ".pptx": pptx_to_md.convert,
    ".xlsx": xlsx_to_md.convert,
    ".xlsm": xlsx_to_md.convert,
    ".csv": csv_to_md.convert,
    ".txt": text_passthrough.convert,
    ".md": text_passthrough.convert,
    ".markdown": text_passthrough.convert,
}


def _worker(src_path_str: str) -> dict:
    src = Path(src_path_str)
    rel = src.relative_to(CORPUS)
    out_path = MARKDOWN_OUT / rel.parent / (rel.name + ".md")
    t0 = time.time()
    try:
        converter = CONVERTER_BY_EXT[src.suffix.lower()]
        text = converter(str(src))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text or "", encoding="utf-8")
        return {
            "source": str(rel).replace("\\", "/"),
            "output": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "status": "empty" if not text or not text.strip() else "ok",
            "chars": len(text) if text else 0,
            "elapsed_sec": round(time.time() - t0, 3),
        }
    except Exception as e:
        return {
            "source": str(rel).replace("\\", "/"),
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "elapsed_sec": round(time.time() - t0, 3),
        }


def _needs_conversion(src: Path, force: bool) -> bool:
    if force:
        return True
    rel = src.relative_to(CORPUS)
    out_path = MARKDOWN_OUT / rel.parent / (rel.name + ".md")
    if not out_path.exists():
        return True
    return src.stat().st_mtime > out_path.stat().st_mtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=None, help="only process the first N files (sample run)")
    ap.add_argument("--force", action="store_true", help="reprocess even if a newer .md already exists")
    ap.add_argument("--ext", type=str, default=None, help="only this extension, e.g. .pdf (for isolated benchmarking/testing)")
    args = ap.parse_args()

    all_files = [p for p in CORPUS.rglob("*") if p.is_file() and p.suffix.lower() in CONVERTER_BY_EXT]
    if args.ext:
        all_files = [p for p in all_files if p.suffix.lower() == args.ext.lower()]

    todo = [p for p in all_files if _needs_conversion(p, args.force)]
    skipped = len(all_files) - len(todo)

    if args.limit:
        todo = todo[: args.limit]

    print(f"Total in-scope files: {len(all_files)} | already converted (skipped): {skipped} | to process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    LOG_PATH.parent.mkdir(exist_ok=True)
    log_f = open(LOG_PATH, "a", encoding="utf-8")

    done = 0
    ok = 0
    failed = 0
    empty = 0
    t_start = time.time()

    with multiprocessing.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_worker, [str(p) for p in todo]):
            done += 1
            if result["status"] == "ok":
                ok += 1
            elif result["status"] == "empty":
                empty += 1
            else:
                failed += 1
                print(f"  ! FAILED: {result['source']} -> {result.get('error')}")

            log_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            log_f.flush()

            if done % 25 == 0 or done == len(todo):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(f"[{done}/{len(todo)}] ok={ok} empty={empty} failed={failed} "
                      f"| {rate:.1f} files/s | ETA {eta/60:.1f} min", flush=True)

    log_f.close()
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min. ok={ok} empty={empty} failed={failed} "
          f"(skipped {skipped} already-converted)")
    print(f"Output: {MARKDOWN_OUT}")
    print(f"Log:    {LOG_PATH}")


if __name__ == "__main__":
    main()

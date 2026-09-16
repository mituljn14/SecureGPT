#!/usr/bin/env python3
"""Time real conversion on a random sample per file type, extrapolate to the full corpus."""
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

random.seed(42)

def sample(ext, n=15):
    files = list(CORPUS.rglob(f"*{ext}"))
    return random.sample(files, min(n, len(files))), len(files)


def bench_pdf(files):
    import pymupdf4llm
    times = []
    for f in files:
        t0 = time.time()
        try:
            pymupdf4llm.to_markdown(str(f))
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        times.append(time.time() - t0)
    return times


def bench_docx(files):
    import docx
    times = []
    for f in files:
        t0 = time.time()
        try:
            d = docx.Document(str(f))
            _ = [p.text for p in d.paragraphs]
            _ = [[c.text for c in r.cells] for t in d.tables for r in t.rows]
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        times.append(time.time() - t0)
    return times


def bench_pptx(files):
    from pptx import Presentation
    times = []
    for f in files:
        t0 = time.time()
        try:
            p = Presentation(str(f))
            for slide in p.slides:
                for shape in slide.shapes:
                    _ = shape.has_text_frame
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        times.append(time.time() - t0)
    return times


def bench_xlsx(files):
    import openpyxl
    times = []
    for f in files:
        t0 = time.time()
        try:
            wb = openpyxl.load_workbook(str(f), data_only=True)
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    pass
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        times.append(time.time() - t0)
    return times


def bench_csv(files):
    times = []
    for f in files:
        t0 = time.time()
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                _ = fh.readlines()
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        times.append(time.time() - t0)
    return times


def report(ext, bench_fn, total_count):
    files, total = sample(ext)
    if not files:
        print(f"{ext}: no files found")
        return
    t0 = time.time()
    times = bench_fn(files)
    wall = time.time() - t0
    if not times:
        print(f"{ext}: all samples failed")
        return
    avg = sum(times) / len(times)
    est_total_sec = avg * total
    print(f"{ext:8s} n_sampled={len(times):3d}/{len(files):3d}  avg={avg:.3f}s/file  "
          f"total_files={total:5d}  est_total={est_total_sec/60:.1f} min "
          f"(single-threaded, sequential)")
    return avg, total


results = {}
results['.pdf'] = report('.pdf', bench_pdf, None)
results['.docx'] = report('.docx', bench_docx, None)
results['.pptx'] = report('.pptx', bench_pptx, None)
results['.xlsx'] = report('.xlsx', bench_xlsx, None)
results['.csv'] = report('.csv', bench_csv, None)

print("\n--- grand total (sequential, single core) ---")
total_min = 0
for ext, r in results.items():
    if r:
        avg, total = r
        total_min += (avg * total) / 60
print(f"Estimated sequential wall time: {total_min:.1f} minutes (~{total_min/60:.2f} hours)")
print(f"With a modest process pool (e.g. 8 workers): ~{total_min/8:.1f} minutes")

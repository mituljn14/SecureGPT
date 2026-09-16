#!/usr/bin/env python3
"""Evidence for the record-per-chunk decision on the big registry CSVs.

Checks (a) whether rows are grouped/sorted by their natural query key --
which decides whether big chunks stay topically coherent or become a blur
of unrelated records -- and (b) renders real sample chunks at 1 / 4 / 10
records per chunk with token counts, so the tradeoff is concrete.
"""
import csv
from collections import Counter
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
ENC = tiktoken.get_encoding("cl100k_base")
tok = lambda t: len(ENC.encode(t, disallowed_special=()))

SRC = ROOT / "corpus/Technology_Digital/CDO/Nova_CDO_Dashboards_Subfolder_Enterprise_Package/Nova_CDO_Dashboards/CSV_Data/Enterprise_Dashboard_Catalog_Part_1.csv"

with open(SRC, encoding="utf-8", errors="ignore", newline="") as f:
    rows = list(csv.reader(f))

header, data = rows[0], rows[1:]
print(f"FILE: {SRC.name}")
print(f"rows={len(data)}  cols={len(header)}")
print(f"columns: {header}\n")

# --- (a) are rows grouped by a natural key, or interleaved? ---
print("=== ROW ORDERING (decides if big chunks stay coherent) ===")
for key in ("Business_Domain", "Dashboard_Type", "Owner"):
    if key not in header:
        continue
    idx = header.index(key)
    vals = [r[idx] for r in data if len(r) > idx]
    distinct = len(set(vals))
    # count adjacent transitions: low = grouped, high = interleaved
    transitions = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    max_trans = len(vals) - 1
    print(f"  {key:18s} distinct={distinct:4d}  adjacent-changes={transitions}/{max_trans} "
          f"({transitions/max_trans:.0%})  -> {'GROUPED' if transitions/max_trans < 0.3 else 'INTERLEAVED'}")
    print(f"     top values: {Counter(vals).most_common(3)}")

# --- (b) render sample chunks at different granularity ---
def render_record(row):
    return "\n".join(f"{h}: {v}" for h, v in zip(header, row) if str(v).strip())

CONTEXT = (
    "Technology_Digital > CDO > Nova_CDO_Dashboards\n"
    "Enterprise_Dashboard_Catalog_Part_1.csv\n"
    "Table: Enterprise Dashboard Catalog (records {a}-{b} of 5000)\n"
    "———\n"
)

print("\n=== SAMPLE CHUNKS AT EACH GRANULARITY ===")
out_lines = []
for n in (1, 4, 10):
    group = data[:n]
    body = "\n\n".join(render_record(r) for r in group)
    chunk = CONTEXT.format(a=1, b=n) + body
    n_chunks_total = -(-len(data) // n)
    domains = {r[header.index("Business_Domain")] for r in group if "Business_Domain" in header}
    print(f"\n--- {n} record(s)/chunk : {tok(chunk):4d} tokens | "
          f"{n_chunks_total:5d} chunks for this ONE file | distinct domains in chunk: {len(domains)} ---")
    preview = chunk[:700]
    print(preview.encode("ascii", "replace").decode())
    if len(chunk) > 700:
        print(f"   ... [+{len(chunk)-700} chars]")

    out_lines.append(f"\n\n{'='*80}\n{n} RECORD(S) PER CHUNK — {tok(chunk)} tokens — "
                     f"{n_chunks_total} chunks for this file\n{'='*80}\n{chunk}")

demo = ROOT / "docs" / "chunk_granularity_samples.txt"
demo.write_text("".join(out_lines), encoding="utf-8")
print(f"\n\nFull side-by-side samples written to: {demo}")

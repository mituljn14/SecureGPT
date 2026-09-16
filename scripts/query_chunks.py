#!/usr/bin/env python3
"""Ad-hoc inspector over chunks.jsonl -- filter by path/department/type/year
and dump readable chunks. Stands in for Qdrant filtering before ingest.

    python scripts/query_chunks.py --path Annual_Report_2018 --limit 12
    python scripts/query_chunks.py --dept Finance --type table_card
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "chunks" / "chunks.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--path", default=None, help="substring match on source_path")
ap.add_argument("--dept", default=None)
ap.add_argument("--type", default=None, help="chunk_type")
ap.add_argument("--year", type=int, default=None)
ap.add_argument("--text", default=None, help="substring match in chunk text")
ap.add_argument("--limit", type=int, default=10)
ap.add_argument("--out", default=None, help="also write full results to this file")
a = ap.parse_args()

hits = []
types = Counter()
for line in open(SRC, encoding="utf-8"):
    c = json.loads(line)
    if a.path and a.path.lower() not in c["source_path"].lower():
        continue
    if a.dept and a.dept.lower() not in (c["department"] or "").lower():
        continue
    if a.type and c["chunk_type"] != a.type:
        continue
    if a.year and c.get("fiscal_year") != a.year:
        continue
    if a.text and a.text.lower() not in c["text"].lower():
        continue
    hits.append(c)
    types[c["chunk_type"]] += 1

print(f"MATCHED {len(hits):,} chunks   types={dict(types)}")
if hits:
    docs = Counter(h["source_path"] for h in hits)
    print(f"across {len(docs)} document(s):")
    for d, n in docs.most_common(8):
        print(f"   {n:5d}  {d}")
print()

shown = hits[: a.limit]
for i, c in enumerate(shown, 1):
    print("=" * 78)
    print(f"[{i}] {c['chunk_type']}  |  {c['token_count']} tok  |  "
          f"heading: {' › '.join(c['heading_path']) or '-'}")
    print(f"    fiscal={c.get('fiscal_year')} Q{c.get('fiscal_quarter')}  "
          f"rows={c.get('row_range')}  ids={c.get('identifiers', [])[:3]}")
    print("-" * 78)
    body = c["embed_text"]
    print(body[:1000] + ("\n   … [truncated]" if len(body) > 1000 else ""))
    print()

if a.out:
    """Readable dump -- the chunk text as it will actually be embedded,
    with only the handful of metadata fields worth eyeballing. The full
    JSON payload is in chunks.jsonl if it is ever needed."""
    with open(ROOT / a.out, "w", encoding="utf-8") as f:
        f.write(f"{len(hits)} CHUNKS\n")
        if hits:
            f.write(f"Document(s): {', '.join(sorted({h['source_path'] for h in hits}))}\n")
        f.write(f"Types: {dict(types)}\n")

        for i, c in enumerate(hits, 1):
            f.write("\n\n" + "=" * 90 + "\n")
            f.write(f"CHUNK {i} of {len(hits)}   [{c['chunk_type']}]   {c['token_count']} tokens\n")
            loc = []
            if c.get("fiscal_year"):
                loc.append(f"FY{c['fiscal_year']}" + (f" Q{c['fiscal_quarter']}" if c.get("fiscal_quarter") else ""))
            if c.get("page"):
                loc.append(f"page {c['page']}")
            if c.get("sheet"):
                loc.append(f"sheet {c['sheet']}")
            if c.get("row_range"):
                loc.append(f"rows {c['row_range'][0]}-{c['row_range'][1]} of {c.get('row_total')}")
            if loc:
                f.write("Location: " + " | ".join(loc) + "\n")
            f.write(f"Section:  {' > '.join(c['heading_path']) or '-'}\n")
            f.write(f"Access:   {', '.join(c['access_roles'])}\n")
            if c.get("identifiers"):
                f.write(f"IDs:      {', '.join(c['identifiers'][:8])}\n")
            f.write("-" * 90 + "\n")
            f.write(c["embed_text"] + "\n")

    print(f"\nWrote all {len(hits)} chunks (readable) to {a.out}")

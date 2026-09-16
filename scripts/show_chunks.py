#!/usr/bin/env python3
"""Dump a diverse, readable sample of produced chunks for human review --
one per (chunk_type, source_format) combination where possible, showing
exactly what gets EMBEDDED plus the key metadata fields."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "chunks" / "chunks.jsonl"
OUT = ROOT / "chunks" / "SAMPLE_CHUNKS.md"

random.seed(7)
buckets = {}
for line in open(SRC, encoding="utf-8"):
    c = json.loads(line)
    key = (c["chunk_type"], c["source_format"])
    buckets.setdefault(key, [])
    if len(buckets[key]) < 40:
        buckets[key].append(c)

# pick a spread of buckets, preferring variety of type then format
wanted = sorted(buckets.keys())
picked = []
for key in wanted:
    picked.append(random.choice(buckets[key]))

KEYS = ["chunk_type", "source_path", "department", "sub_department", "access_roles",
        "sensitivity", "region", "heading_path", "section_title", "fiscal_year",
        "fiscal_quarter", "doc_date", "page", "sheet", "slide", "row_range",
        "row_total", "table_caption", "identifiers", "token_count",
        "content_hash", "prev_chunk_id", "next_chunk_id", "parent_block_id"]

lines = ["# Sample chunks produced by the pipeline\n",
         f"\nOne example per (chunk_type × source_format) combination — {len(picked)} shown.\n",
         "\n`EMBEDDED TEXT` is what BGE-M3 actually sees (context header + body).\n",
         "The clean `text` field, without the header, is what the LLM gets for citation.\n"]

for i, c in enumerate(picked, 1):
    lines.append(f"\n\n---\n\n## {i}. `{c['chunk_type']}` from `.{c['source_format']}`\n")
    lines.append(f"\n**File:** `{c['source_path']}`\n")
    lines.append("\n### EMBEDDED TEXT\n\n```\n")
    body = c["embed_text"]
    lines.append(body[:1400] + ("\n… [truncated]" if len(body) > 1400 else ""))
    lines.append("\n```\n")
    lines.append("\n### METADATA\n\n```json\n")
    meta = {k: c.get(k) for k in KEYS if c.get(k) not in (None, [], "")}
    lines.append(json.dumps(meta, indent=2, ensure_ascii=False))
    lines.append("\n```\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"Wrote {len(picked)} sample chunks to {OUT}")
for i, c in enumerate(picked, 1):
    print(f"  {i:2d}. {c['chunk_type']:14s} .{c['source_format']:5s} {c['token_count']:4d} tok  {c['source_path'][:70]}")

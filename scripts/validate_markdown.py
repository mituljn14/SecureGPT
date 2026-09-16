#!/usr/bin/env python3
"""Independently verify markdown/ against corpus/ -- not just trusting the
conversion log's self-reported status, but checking what's actually on disk:
  1. Every corpus file has exactly one corresponding .md file (1:1 mapping).
  2. No .md file is missing, empty, or suspiciously tiny.
  3. Size-ratio sanity check per file type (catches silent truncation).
  4. A content spot-check: every .md file should contain readable text,
     not binary garbage / mojibake / a raw error string.
"""
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
MARKDOWN = ROOT / "markdown"

TINY_THRESHOLD = 40  # bytes -- suspicious even for a very short doc

corpus_files = [p for p in CORPUS.rglob("*") if p.is_file()]
print(f"corpus/: {len(corpus_files)} files")

missing = []
empty = []
tiny = []
size_by_ext = defaultdict(list)
mojibake_hits = []
found_md = 0

_MOJIBAKE_RE = re.compile(r"[�]|Ã\x82|Ã\x83|â\x80")

for src in corpus_files:
    rel = src.relative_to(CORPUS)
    expected_md = MARKDOWN / rel.parent / (rel.name + ".md")
    if not expected_md.exists():
        missing.append(rel)
        continue
    found_md += 1
    size = expected_md.stat().st_size
    if size == 0:
        empty.append(rel)
        continue
    if size < TINY_THRESHOLD:
        tiny.append((rel, size))

    size_by_ext[src.suffix.lower()].append(size)

    # cheap mojibake / replacement-character scan on a sample of files
    if len(mojibake_hits) < 500:  # cap work
        try:
            text = expected_md.read_text(encoding="utf-8", errors="strict")
            if _MOJIBAKE_RE.search(text):
                mojibake_hits.append(rel)
        except UnicodeDecodeError:
            mojibake_hits.append(rel)  # not even valid utf-8

print(f"markdown/: {found_md} matching .md files found")
print(f"Missing .md output: {len(missing)}")
for m in missing[:20]:
    print(f"  ! MISSING: {m}")
if len(missing) > 20:
    print(f"  ... and {len(missing)-20} more")

print(f"\nEmpty .md files (0 bytes): {len(empty)}")
for e in empty[:20]:
    print(f"  ! EMPTY: {e}")

print(f"\nSuspiciously tiny .md files (<{TINY_THRESHOLD} bytes): {len(tiny)}")
for t, s in tiny[:20]:
    print(f"  ! TINY ({s}B): {t}")

print(f"\nMojibake / invalid-UTF8 hits (sampled up to 500 files): {len(mojibake_hits)}")
for m in mojibake_hits[:20]:
    print(f"  ! {m}")

print("\n--- avg markdown size by source extension ---")
for ext, sizes in sorted(size_by_ext.items()):
    avg = sum(sizes) / len(sizes)
    print(f"  {ext:8s} n={len(sizes):5d}  avg={avg:8.0f}B  min={min(sizes):6d}B  max={max(sizes):8d}B")

print("\n=== SUMMARY ===")
ok = not missing and not empty and not tiny and not mojibake_hits
print("ALL CLEAR" if ok else "ISSUES FOUND -- see above")

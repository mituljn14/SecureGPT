#!/usr/bin/env python3
"""One-off in-place repair of malformed markdown tables in markdown/,
using pipeline.converters.md_normalize. Avoids re-running the expensive
PDF conversion just to fix a handful of files: only rewrites a file if
normalization actually changed it."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.converters.md_normalize import normalize_tables

MARKDOWN = ROOT / "markdown"

changed = 0
scanned = 0
for md in MARKDOWN.rglob("*.md"):
    scanned += 1
    original = md.read_text(encoding="utf-8", errors="replace")
    fixed = normalize_tables(original)
    if fixed != original:
        md.write_text(fixed, encoding="utf-8")
        changed += 1
        if changed <= 40:
            print(f"  repaired: {md.relative_to(MARKDOWN)}")

print(f"\nScanned {scanned} files, repaired {changed}.")

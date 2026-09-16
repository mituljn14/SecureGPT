#!/usr/bin/env python3
"""Fast integrity scan: check magic bytes of every in-scope file, and
detect duplicate '(1)'/'(2)' suffixed sibling folders across the corpus."""
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

MAGIC = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".xlsx": (b"PK\x03\x04",),
}

bad = []
checked = 0
for ext, magics in MAGIC.items():
    for f in CORPUS.rglob(f"*{ext}"):
        checked += 1
        try:
            with open(f, "rb") as fh:
                head = fh.read(8)
        except Exception as e:
            bad.append((f, f"unreadable: {e}"))
            continue
        if not any(head.startswith(m) for m in magics):
            bad.append((f, f"bad magic bytes: {head[:16]!r}"))

print(f"Checked {checked} files (pdf/docx/pptx/xlsx magic-byte validity)")
print(f"Corrupted/invalid: {len(bad)}")
for f, reason in bad:
    print(f"  ! {f.relative_to(CORPUS)}  -- {reason}")

# duplicate "(1)"/"(2)" sibling folder detection
dup_re = re.compile(r"^(.*) \((\d+)\)$")
dirs_by_parent = defaultdict(set)
for d in CORPUS.rglob("*"):
    if d.is_dir():
        dirs_by_parent[d.parent].add(d.name)

dup_pairs = []
for parent, names in dirs_by_parent.items():
    for name in names:
        m = dup_re.match(name)
        if m and m.group(1) in names:
            dup_pairs.append(parent / name)

print(f"\nDuplicate-suffixed folders ('(1)', '(2)', ... with a same-named original sibling): {len(dup_pairs)}")
for p in sorted(dup_pairs)[:40]:
    print(f"  {p.relative_to(CORPUS)}")
if len(dup_pairs) > 40:
    print(f"  ... and {len(dup_pairs) - 40} more")

#!/usr/bin/env python3
"""
Deep content-level validation of markdown/ -- goes past "file exists and is
non-empty" into whether the CONTENT is structurally sound and chunkable:

  1. Markdown table integrity: header / separator / body row column counts
     must agree. A mismatch means a broken grid that will mangle retrieval.
  2. Unclosed code fences (would swallow the rest of a doc into one block).
  3. Content-bearing check: is there real text, or only headings/whitespace?
  4. Truncation markers / error strings leaked into output.
  5. Repeated-block detection (same paragraph emitted many times = converter loop).
  6. Token statistics per file type -- this sizes the chunking + embedding job.
  7. Per-department inventory.
  8. Specific audit of files that came from source .md (passthrough path).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
MARKDOWN = ROOT / "markdown"

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    count_tokens = lambda t: len(_ENC.encode(t, disallowed_special=()))
except Exception:
    count_tokens = lambda t: int(len(t.split()) / 0.75) + 1

TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEP_ROW_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
ERROR_MARKERS = re.compile(r"(Traceback \(most recent call last\)|<\?xml version|%PDF-|PK\x03\x04|\[ERROR\]|NoneType)", re.I)


def split_cells(line: str):
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    # split on unescaped pipes
    return re.split(r"(?<!\\)\|", inner)


def analyse(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    issues = []

    # --- code fences ---
    if text.count("```") % 2 != 0:
        issues.append("unclosed-code-fence")

    # --- error markers leaked into content ---
    if ERROR_MARKERS.search(text):
        issues.append("error-marker-in-content")

    # --- table integrity ---
    lines = text.split("\n")
    n_tables = 0
    malformed = 0
    i = 0
    while i < len(lines):
        if TABLE_ROW_RE.match(lines[i]) and i + 1 < len(lines) and SEP_ROW_RE.match(lines[i + 1]):
            n_tables += 1
            header_cols = len(split_cells(lines[i]))
            sep_cols = len(split_cells(lines[i + 1]))
            bad = header_cols != sep_cols
            j = i + 2
            while j < len(lines) and TABLE_ROW_RE.match(lines[j]):
                if len(split_cells(lines[j])) != header_cols:
                    bad = True
                j += 1
            if bad:
                malformed += 1
            i = j
        else:
            i += 1
    if malformed:
        issues.append(f"malformed-tables:{malformed}")

    # --- content-bearing: strip headings/table markup, is anything left? ---
    body = "\n".join(
        l for l in lines
        if l.strip() and not l.strip().startswith("#") and not TABLE_ROW_RE.match(l)
    )
    if n_tables == 0 and len(body.strip()) < 50:
        issues.append("no-substantive-content")

    # --- repeated block detection (converter loop) ---
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 120]
    if paras:
        counts = defaultdict(int)
        for p in paras:
            counts[p] += 1
        worst = max(counts.values())
        if worst >= 5 and worst / len(paras) > 0.3:
            issues.append(f"repeated-block-x{worst}")

    return {
        "tokens": count_tokens(text),
        "chars": len(text),
        "n_tables": n_tables,
        "issues": issues,
    }


def main():
    md_files = sorted(MARKDOWN.rglob("*.md"))
    print(f"Analysing {len(md_files)} markdown files (deep content check)...\n")

    by_ext = defaultdict(lambda: {"n": 0, "tokens": 0, "tables": 0, "issues": 0})
    by_dept = defaultdict(lambda: {"n": 0, "tokens": 0})
    all_issues = []
    token_sizes = []
    total_tokens = 0
    total_tables = 0

    for i, md in enumerate(md_files, 1):
        if i % 1000 == 0:
            print(f"  ...{i}/{len(md_files)}")
        rel = md.relative_to(MARKDOWN)
        # source extension is the segment before the trailing .md
        src_name = md.name[:-3]
        src_ext = Path(src_name).suffix.lower() or ".unknown"
        dept = rel.parts[0] if rel.parts else "?"

        r = analyse(md)
        total_tokens += r["tokens"]
        total_tables += r["n_tables"]
        token_sizes.append(r["tokens"])

        e = by_ext[src_ext]
        e["n"] += 1
        e["tokens"] += r["tokens"]
        e["tables"] += r["n_tables"]
        if r["issues"]:
            e["issues"] += 1
            all_issues.append((rel, r["issues"]))

        d = by_dept[dept]
        d["n"] += 1
        d["tokens"] += r["tokens"]

    print("\n=== PER SOURCE TYPE ===")
    print(f"{'ext':10s} {'files':>6s} {'tokens':>12s} {'avg tok':>9s} {'tables':>8s} {'w/issues':>9s}")
    for ext, s in sorted(by_ext.items(), key=lambda x: -x[1]["tokens"]):
        print(f"{ext:10s} {s['n']:6d} {s['tokens']:12,d} {s['tokens']//max(s['n'],1):9,d} {s['tables']:8d} {s['issues']:9d}")

    print("\n=== PER DEPARTMENT ===")
    for dept, s in sorted(by_dept.items(), key=lambda x: -x[1]["tokens"]):
        print(f"  {dept:24s} {s['n']:5d} files  {s['tokens']:12,d} tokens")

    token_sizes.sort()
    def pct(p):
        return token_sizes[min(int(len(token_sizes) * p), len(token_sizes) - 1)]

    print("\n=== CORPUS SIZE (drives chunking + embedding cost) ===")
    print(f"  total tokens        : {total_tokens:,}")
    print(f"  total markdown tables: {total_tables:,}")
    print(f"  per-file tokens  p50: {pct(0.50):,}  p90: {pct(0.90):,}  p99: {pct(0.99):,}  max: {token_sizes[-1]:,}")
    for target in (512, 800):
        print(f"  est. chunks @ ~{target} tok (with ~12% overlap): ~{int(total_tokens / target * 1.12):,}")

    print(f"\n=== CONTENT ISSUES: {len(all_issues)} files ===")
    kinds = defaultdict(int)
    for rel, issues in all_issues:
        for iss in issues:
            kinds[iss.split(":")[0]] += 1
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    for rel, issues in all_issues[:25]:
        print(f"    ! {rel}  -> {issues}")
    if len(all_issues) > 25:
        print(f"    ... and {len(all_issues)-25} more")

    with open(ROOT / "docs" / "deep_validation_issues.json", "w", encoding="utf-8") as f:
        json.dump([{"file": str(r), "issues": i} for r, i in all_issues], f, indent=2)

    print("\n=== SOURCE .md PASSTHROUGH AUDIT ===")
    src_md = sorted(CORPUS.rglob("*.md"))
    print(f"  source .md files in corpus/: {len(src_md)}")
    missing = [p for p in src_md
               if not (MARKDOWN / p.relative_to(CORPUS).parent / (p.name + ".md")).exists()]
    print(f"  converted into markdown/  : {len(src_md) - len(missing)}")
    for m in missing[:10]:
        print(f"    ! MISSING: {m.relative_to(CORPUS)}")

    print("\n" + ("ALL CLEAR" if not all_issues else f"REVIEW {len(all_issues)} FILES ABOVE"))


if __name__ == "__main__":
    main()

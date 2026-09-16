#!/usr/bin/env python3
"""
Phase 0 — unzip, recursively unpack nested zips, classify every file into
IN-SCOPE (goes into ./corpus for processing) vs OUT-OF-SCOPE (documented,
left out for now), and write a manifest + human-readable scope report.

Run from the "Data zip" project root.
"""
import csv
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_ZIP = ROOT / "data.zip"
RAW_DIR = ROOT / "raw"          # full extracted mirror (nested zips unpacked in place)
CORPUS_DIR = ROOT / "corpus"    # filtered copy: only in-scope files, same relative layout
DOCS_DIR = ROOT / "docs"

DOCS_DIR.mkdir(exist_ok=True)

# --- extension classification --------------------------------------------------

IN_SCOPE = {
    ".pdf": "core business document",
    ".docx": "core business document",
    ".xlsx": "core business document (tabular)",
    ".xlsm": "core business document (tabular, macro-enabled)",
    ".csv": "core business document (tabular)",
    ".pptx": "core business document (slides)",
    ".txt": "plain text document",
    ".md": "plain markdown document",
    ".markdown": "plain markdown document",
}

OUT_OF_SCOPE = {
    ".py": "source code — not organizational knowledge prose; revisit if CTO/CIO runbooks need indexing",
    ".java": "source code — same as .py",
    ".cpp": "source code — same as .py",
    ".js": "source code — same as .py",
    ".sql": "DB schema / query scripts — same reasoning as source code",
    ".msg": "Outlook email — needs a dedicated .msg parser (extract-msg); deferred, not skipped forever",
    ".png": "image/diagram — needs OCR or a vision captioner to be searchable; deferred",
    ".svg": "vector image/diagram — same as .png",
    ".xml": "technical config, not business prose/tabular knowledge; deferred",
    ".json": "structured config/data — ambiguous (could be real business data); deferred pending your review",
    ".yaml": "structured config/data — same ambiguity as .json; deferred pending your review",
    ".yml": "structured config/data — same ambiguity as .json; deferred pending your review",
}

JUNK_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}


def extract_nested_zips(base: Path) -> int:
    """Recursively find .zip files under base, extract them next to
    themselves into a same-named folder, then delete the archive.
    Repeats until no nested zips remain (zips inside zips)."""
    extracted = 0
    while True:
        zips = list(base.rglob("*.zip"))
        if not zips:
            break
        for zp in zips:
            target_dir = zp.with_suffix("")  # e.g. "files (15).zip" -> "files (15)"
            target_dir.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(target_dir)
                zp.unlink()
                extracted += 1
            except Exception as e:
                print(f"  ! failed to extract {zp}: {e}")
    return extracted


def main():
    # 1. extract top-level data.zip
    if RAW_DIR.exists():
        print(f"raw/ already exists at {RAW_DIR} — skipping top-level extraction, "
              f"reusing it (delete raw/ manually to force a clean re-extract).")
    else:
        print(f"Extracting {SRC_ZIP.name} -> {RAW_DIR} ...")
        with zipfile.ZipFile(SRC_ZIP, "r") as zf:
            zf.extractall(RAW_DIR)

    # 2. recursively unpack nested zips (the 26 "files (N).zip" / "PhaseN-*.zip")
    print("Unpacking nested zip files...")
    n = extract_nested_zips(RAW_DIR)
    print(f"  unpacked {n} nested zip archives")

    # 3. remove OS junk
    junk_removed = 0
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.name.lower() in JUNK_NAMES:
            p.unlink()
            junk_removed += 1
    print(f"  removed {junk_removed} OS junk files")

    # 4. classify + copy in-scope files into corpus/, build manifest
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir()

    manifest_rows = []
    scope_counts = {}
    excluded_counts = {}
    unclassified = {}

    data_root = RAW_DIR / "data"
    walk_root = data_root if data_root.exists() else RAW_DIR

    for p in walk_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        rel = p.relative_to(walk_root)

        if ext in IN_SCOPE:
            dest = CORPUS_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            scope_counts[ext] = scope_counts.get(ext, 0) + 1
            manifest_rows.append([str(rel), ext, "IN_SCOPE", IN_SCOPE[ext]])
        elif ext in OUT_OF_SCOPE:
            excluded_counts[ext] = excluded_counts.get(ext, 0) + 1
            manifest_rows.append([str(rel), ext, "OUT_OF_SCOPE", OUT_OF_SCOPE[ext]])
        else:
            unclassified[ext] = unclassified.get(ext, 0) + 1
            manifest_rows.append([str(rel), ext, "UNCLASSIFIED", "extension not seen during scoping — review manually"])

    # 5. write manifest.csv
    manifest_path = DOCS_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "extension", "status", "reason"])
        w.writerows(manifest_rows)

    # 6. write SCOPE.md
    total_in = sum(scope_counts.values())
    total_out = sum(excluded_counts.values())
    total_unclass = sum(unclassified.values())

    lines = []
    lines.append("# Data Scope Decisions\n")
    lines.append(f"Generated from `data.zip` ({total_in + total_out + total_unclass} files walked).\n")
    lines.append(f"\n## IN SCOPE — copied to `corpus/` ({total_in} files)\n")
    lines.append("| Extension | Count | Reason |\n|---|---|---|\n")
    for ext, cnt in sorted(scope_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {ext} | {cnt} | {IN_SCOPE[ext]} |\n")

    lines.append(f"\n## OUT OF SCOPE — deferred, left in `raw/` only ({total_out} files)\n")
    lines.append("| Extension | Count | Reason |\n|---|---|---|\n")
    for ext, cnt in sorted(excluded_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {ext} | {cnt} | {OUT_OF_SCOPE[ext]} |\n")

    if unclassified:
        lines.append(f"\n## UNCLASSIFIED — needs a decision ({total_unclass} files)\n")
        lines.append("| Extension | Count |\n|---|---|\n")
        for ext, cnt in sorted(unclassified.items(), key=lambda x: -x[1]):
            lines.append(f"| {ext or '(no extension)'} | {cnt} |\n")

    lines.append("\nFull per-file breakdown: `docs/manifest.csv`\n")

    with open(DOCS_DIR / "SCOPE.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\n=== DONE ===")
    print(f"IN SCOPE:        {total_in} files -> {CORPUS_DIR}")
    print(f"OUT OF SCOPE:    {total_out} files (documented, not copied)")
    print(f"UNCLASSIFIED:    {total_unclass} files")
    print(f"Manifest:        {manifest_path}")
    print(f"Scope report:    {DOCS_DIR / 'SCOPE.md'}")


if __name__ == "__main__":
    main()

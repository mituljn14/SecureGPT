#!/usr/bin/env python3
"""
Phase 0b — clean corpus/ in place:
  1. Delete files that are corrupted/placeholder (bad magic bytes for their
     extension, or genuinely empty).
  2. De-duplicate by content hash (sha256) across the WHOLE corpus, not just
     the one folder we spotted manually — keeps the first occurrence
     (shortest relative path, i.e. prefers the non-"(1)"-suffixed copy when
     there's a tie), deletes the rest.

Only touches corpus/. raw/ (the untouched extraction) and data.zip (the
original archive) are never modified — corpus/ can always be rebuilt from
them via scripts/scope_and_extract.py.

Writes docs/CLEANUP_LOG.md with the full list of what was removed and why,
and appends a summary paragraph to README.md.
"""
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
DOCS = ROOT / "docs"
README = ROOT / "README.md"

MAGIC = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".xlsx": (b"PK\x03\x04",),
}


def is_corrupted(f: Path) -> str | None:
    """Return a reason string if corrupted, else None."""
    try:
        size = f.stat().st_size
    except OSError as e:
        return f"unreadable: {e}"
    if size == 0:
        return "empty file (0 bytes)"

    ext = f.suffix.lower()
    if ext in MAGIC:
        try:
            with open(f, "rb") as fh:
                head = fh.read(8)
        except Exception as e:
            return f"unreadable: {e}"
        if not any(head.startswith(m) for m in MAGIC[ext]):
            return f"invalid {ext} content (bad magic bytes: {head[:16]!r} — likely a placeholder/corrupted file)"
    return None


def sha256_of(f: Path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    all_files = [p for p in CORPUS.rglob("*") if p.is_file()]
    print(f"Scanning {len(all_files)} files in corpus/ ...")

    # 1. corrupted / empty
    removed_corrupted = []
    remaining = []
    for f in all_files:
        reason = is_corrupted(f)
        if reason:
            removed_corrupted.append((f, reason))
        else:
            remaining.append(f)

    for f, reason in removed_corrupted:
        f.unlink()

    print(f"Removed {len(removed_corrupted)} corrupted/empty files")

    # 2. content-hash dedup across whatever's left
    print("Hashing remaining files for dedup...")
    by_hash = {}
    for f in remaining:
        try:
            h = sha256_of(f)
        except Exception as e:
            print(f"  ! could not hash {f}: {e}")
            continue
        by_hash.setdefault(h, []).append(f)

    removed_dupes = []
    for h, files in by_hash.items():
        if len(files) <= 1:
            continue
        files.sort(key=lambda p: (len(str(p)), str(p)))  # keep shortest/cleanest path
        keep, dupes = files[0], files[1:]
        for d in dupes:
            removed_dupes.append((d, keep))
            d.unlink()

    print(f"Removed {len(removed_dupes)} duplicate files (identical content to another kept file)")

    # remove now-empty directories left behind
    removed_dirs = 0
    for d in sorted(CORPUS.rglob("*"), key=lambda p: -len(str(p))):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed_dirs += 1
    print(f"Removed {removed_dirs} now-empty directories")

    # --- write CLEANUP_LOG.md ---
    lines = ["# Cleanup Log\n", f"Run at {datetime.now(timezone.utc).isoformat()}\n"]
    lines.append(f"\n## Corrupted / empty files removed from `corpus/` ({len(removed_corrupted)})\n")
    lines.append("These were deleted from `corpus/` only — the originals are untouched in `raw/` and `data.zip`.\n\n")
    lines.append("| File | Reason |\n|---|---|\n")
    for f, reason in removed_corrupted:
        lines.append(f"| {f.relative_to(CORPUS)} | {reason} |\n")

    lines.append(f"\n## Duplicate files removed from `corpus/` ({len(removed_dupes)})\n")
    lines.append("Identical content (sha256 match) to a kept file elsewhere in the corpus.\n\n")
    lines.append("| Removed | Kept (identical content) |\n|---|---|\n")
    for d, keep in removed_dupes:
        lines.append(f"| {d.relative_to(CORPUS)} | {keep.relative_to(CORPUS)} |\n")

    with open(DOCS / "CLEANUP_LOG.md", "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    # --- append paragraph to README.md ---
    final_count = len(all_files) - len(removed_corrupted) - len(removed_dupes)
    readme_para = f"""
## Corpus cleanup ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})

`corpus/` started as a straight filtered copy of `raw/` (which is itself the untouched
extraction of `data.zip`). A cleanup pass then ran against `corpus/` only:

- **{len(removed_corrupted)} corrupted/empty files deleted** — files whose content didn't
  match their extension's real file signature (e.g. a `.xlsx` that was actually just its own
  filename typed as text) or were 0 bytes. Mostly concentrated in
  `Legal_Compliance/CLO/*_Repository/` — see `docs/DATA_QUALITY_ISSUES.md`.
- **{len(removed_dupes)} duplicate files deleted** — exact content matches (sha256) of
  another file kept elsewhere in the corpus, e.g. the `Export_Control_Repository (1)/`
  folder duplicating `Export_Control_Repository/`.
- Corpus went from {len(all_files)} → **{final_count} files**.

**Nothing was deleted from the original data.** `raw/` (full extraction) and `data.zip`
(original archive) are untouched — if you ever want any of the removed files back, they're
still there. Full removal list with reasons: `docs/CLEANUP_LOG.md`.
"""
    with open(README, "a", encoding="utf-8") as fh:
        fh.write(readme_para)

    print(f"\nFinal corpus size: {final_count} files (was {len(all_files)})")
    print(f"Log: {DOCS / 'CLEANUP_LOG.md'}")
    print(f"README.md updated: {README}")


if __name__ == "__main__":
    main()

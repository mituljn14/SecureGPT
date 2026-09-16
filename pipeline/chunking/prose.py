# pipeline/chunking/prose.py
"""
Stage 2b — prose routing.

Heading-scoped recursive splitting. Deliberately NOT semantic/embedding
based: benchmarks (NAACL 2025 "Is Semantic Chunking Worth the
Computational Cost?"; Feb-2026 7-strategy comparison where recursive
512-token ranked #1 at 69% vs semantic at 54%) show embedding-similarity
merging costs far more and retrieves worse than structure-first splitting
on well-formed documents. Our markdown is well-formed by construction, so
headings and paragraphs are a stronger and free signal.

Order of preference when a unit is too large:
    whole section -> paragraphs -> sentences -> hard token window
Never splits mid-sentence when a sentence boundary is available.
"""
import re
from typing import List

from .tokens import count_tokens, decode, encode

TARGET_TOKENS = 512
MAX_TOKENS = 1024
MIN_TOKENS = 128
OVERLAP_TOKENS = 64

_PARA_RE = re.compile(r"\n\s*\n")
# Split after . ! ? when followed by whitespace + capital/quote/digit, but not
# after common abbreviations or a single initial.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Pvt|Co|Corp|vs|etc|e\.g|i\.e|No|Fig|Sec|Rev)\.$")


def split_sentences(text: str) -> List[str]:
    raw = _SENT_RE.split(text)
    out: List[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        # re-join a fragment that was split on an abbreviation
        if out and _ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return out


def _hard_window(text: str, size: int = TARGET_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    ids = encode(text)
    if not ids:
        return []
    step = max(1, size - overlap)
    out: List[str] = []
    i = 0
    while i < len(ids):
        piece = decode(ids[i: i + size]).strip()
        if piece:
            out.append(piece)
        if i + size >= len(ids):
            break
        i += step
    return out


def _pack(units: List[str], target: int = TARGET_TOKENS) -> List[str]:
    """Greedily pack whole units (paragraphs or sentences) up to target."""
    packed: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    for unit in units:
        ut = count_tokens(unit)
        if ut > MAX_TOKENS:
            if buf:
                packed.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            packed.extend(_hard_window(unit))
            continue
        if buf and buf_tokens + ut > target:
            packed.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(unit)
        buf_tokens += ut

    if buf:
        packed.append("\n\n".join(buf))
    return packed


def _add_overlap(chunks: List[str], overlap: int = OVERLAP_TOKENS) -> List[str]:
    """Prepend the tail of the previous chunk, snapped to a sentence
    boundary so we never start a chunk mid-sentence."""
    if len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        sents = split_sentences(prev)
        tail: List[str] = []
        total = 0
        for s in reversed(sents):
            st = count_tokens(s)
            if total + st > overlap:
                break
            tail.insert(0, s)
            total += st
        out.append((" ".join(tail) + "\n\n" + cur).strip() if tail else cur)
    return out


def chunk_prose(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if count_tokens(text) <= TARGET_TOKENS:
        return [text]                                   # section fits: one chunk

    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    units: List[str] = []
    for para in paragraphs:
        if count_tokens(para) <= TARGET_TOKENS:
            units.append(para)
        else:
            units.extend(split_sentences(para) or [para])

    chunks = _pack(units)
    chunks = merge_small(chunks)
    return _add_overlap(chunks)


def merge_small(chunks: List[str], min_tokens: int = MIN_TOKENS) -> List[str]:
    """Fold undersized chunks forward; if the last one is still short,
    fold it backward instead."""
    if len(chunks) < 2:
        return chunks
    merged: List[str] = []
    buf = chunks[0]
    for nxt in chunks[1:]:
        if count_tokens(buf) < min_tokens and count_tokens(buf) + count_tokens(nxt) <= MAX_TOKENS:
            buf = buf + "\n\n" + nxt
        else:
            merged.append(buf)
            buf = nxt
    merged.append(buf)
    if len(merged) > 1 and count_tokens(merged[-1]) < min_tokens:
        tail = merged.pop()
        merged[-1] = merged[-1] + "\n\n" + tail
    return merged

#!/usr/bin/env python3
"""
Hybrid retrieval against securegpt_v2.

    python pipeline/search.py "what was EBITDA margin in 2018?"
    python pipeline/search.py "who owns the warranty analytics dashboard?" --role CDO
    python pipeline/search.py "revenue variance Germany" --year 2027 --type table_rows

Implements the retrieval design end to end:

  dense  (BGE-M3 semantic)  ─┐
                             ├─ RRF fusion, server-side, one round trip
  sparse (BGE-M3 lexical)   ─┘
        ↓
  RBAC filter applied INSIDE the query (never post-filtered in Python)
        ↓
  duplicate collapse by content_hash
        ↓
  optional neighbour / table expansion  (the no-data-loss step)

The reranker slot sits between fusion and expansion and is intentionally
left empty for now.
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLLECTION = os.getenv("QDRANT_COLLECTION", "securegpt_v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_model = None


def get_model():
    """BGE-M3 on CPU is fine for a single query (~100ms)."""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    return _model


def embed_query(text: str):
    out = get_model().encode([text], batch_size=1, max_length=512,
                             return_dense=True, return_sparse=True)
    dense = out["dense_vecs"][0].tolist()
    lw = out["lexical_weights"][0]
    sparse = qm.SparseVector(indices=[int(k) for k in lw.keys()],
                             values=[float(v) for v in lw.values()])
    return dense, sparse


def build_filter(role: Optional[str] = None, department: Optional[str] = None,
                 year: Optional[int] = None, quarter: Optional[int] = None,
                 chunk_type: Optional[str] = None, doc_id: Optional[str] = None,
                 identifier: Optional[str] = None) -> Optional[qm.Filter]:
    must = []
    # RBAC first: this is a security boundary, so it is always part of the
    # query Qdrant executes, never a filter applied to results afterwards.
    if role:
        must.append(qm.FieldCondition(key="access_roles", match=qm.MatchAny(any=[role])))
    if department:
        must.append(qm.FieldCondition(key="department", match=qm.MatchValue(value=department)))
    if year:
        must.append(qm.FieldCondition(key="fiscal_year", match=qm.MatchValue(value=year)))
    if quarter:
        must.append(qm.FieldCondition(key="fiscal_quarter", match=qm.MatchValue(value=quarter)))
    if chunk_type:
        must.append(qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value=chunk_type)))
    if doc_id:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
    if identifier:
        must.append(qm.FieldCondition(key="identifiers", match=qm.MatchAny(any=[identifier])))
    return qm.Filter(must=must) if must else None


def hybrid_search(client, query: str, *, limit: int = 10, prefetch: int = 100,
                  qfilter: Optional[qm.Filter] = None):
    dense, sparse = embed_query(query)
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            qm.Prefetch(query=dense, using="dense", limit=prefetch, filter=qfilter),
            qm.Prefetch(query=sparse, using="sparse", limit=prefetch, filter=qfilter),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
    )
    return res.points


def collapse_duplicates(points):
    """Source data repeats boilerplate heavily (one paragraph occurs 96x in a
    single PDF). Without this, identical text can occupy every top-k slot."""
    seen, out = set(), []
    for p in points:
        h = p.payload.get("content_hash")
        if h and h in seen:
            continue
        if h:
            seen.add(h)
        out.append(p)
    return out


def expand_context(client, points, max_expand: int = 3):
    """No-data-loss step: pull neighbours for prose and, for a table slice,
    every sibling slice of the same table via parent_block_id."""
    extra_ids, table_blocks = [], []
    for p in points[:max_expand]:
        pl = p.payload
        for key in ("prev_chunk_id", "next_chunk_id"):
            if pl.get(key):
                extra_ids.append(pl[key])
        if pl.get("chunk_type") in ("table_rows", "table_record") and pl.get("parent_block_id"):
            table_blocks.append(pl["parent_block_id"])

    extras = []
    if extra_ids:
        extras += client.retrieve(collection_name=COLLECTION, ids=extra_ids, with_payload=True)
    for blk in table_blocks[:1]:
        got, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(
                key="parent_block_id", match=qm.MatchValue(value=blk))]),
            limit=50, with_payload=True)
        extras += got
    return extras


def format_hit(i, p):
    pl = p.payload
    loc = []
    if pl.get("fiscal_year"):
        loc.append(f"FY{pl['fiscal_year']}" + (f" Q{pl['fiscal_quarter']}" if pl.get("fiscal_quarter") else ""))
    if pl.get("page"):
        loc.append(f"p.{pl['page']}")
    if pl.get("sheet"):
        loc.append(f"sheet {pl['sheet']}")
    if pl.get("row_range"):
        loc.append(f"rows {pl['row_range'][0]}-{pl['row_range'][1]}")
    head = (f"[{i}] score={getattr(p,'score',0):.4f}  {pl.get('chunk_type')}  "
            f"{'  |  '.join(loc)}")
    src = f"    {pl.get('department')}/{pl.get('sub_department')}  ·  {pl.get('file_name')}"
    sec = f"    section: {' > '.join(pl.get('heading_path') or []) or '-'}"
    body = (pl.get("text") or "")[:600]
    return f"{head}\n{src}\n{sec}\n    {'-'*70}\n" + "\n".join("    " + l for l in body.split("\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--role"); ap.add_argument("--dept"); ap.add_argument("--type")
    ap.add_argument("--year", type=int); ap.add_argument("--quarter", type=int)
    ap.add_argument("--id", dest="identifier")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--expand", action="store_true", help="show neighbour/table expansion")
    a = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    f = build_filter(role=a.role, department=a.dept, year=a.year,
                     quarter=a.quarter, chunk_type=a.type, identifier=a.identifier)

    pts = hybrid_search(client, a.query, limit=a.limit * 3, qfilter=f)
    pts = collapse_duplicates(pts)[: a.limit]

    print(f"\nQUERY: {a.query}")
    filt = {k: v for k, v in vars(a).items() if v and k not in ("query", "limit", "expand")}
    print(f"FILTERS: {filt or 'none'}   ->  {len(pts)} results\n")
    for i, p in enumerate(pts, 1):
        print(format_hit(i, p)); print()

    if a.expand and pts:
        extras = expand_context(client, pts)
        print(f"--- context expansion: {len(extras)} additional chunks "
              f"(neighbours + full table) ---")
        for e in extras[:5]:
            print(f"    + {e.payload.get('chunk_type')}  "
                  f"rows={e.payload.get('row_range')}  "
                  f"{(e.payload.get('text') or '')[:120]}...")


if __name__ == "__main__":
    main()

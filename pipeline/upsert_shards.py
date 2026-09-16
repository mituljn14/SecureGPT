#!/usr/bin/env python3
"""
Join embedded vectors (from the RunPod shards) back to the full metadata
that never left this machine, and upsert into local Qdrant.

    python pipeline/upsert_shards.py

MEMORY DESIGN -- this is the whole point of the rewrite.
The first version built an id -> payload map for all 763,497 chunks up
front (~5GB) and was OOM-killed alongside Qdrant's own 3.9GB. Shard K is
exactly lines [K*SHARD_SIZE, K*SHARD_SIZE+n) of chunks.jsonl (verified:
embed_input.jsonl.gz was written by streaming chunks.jsonl in order, and
the pod sliced ids sequentially), so we stream the file ONCE and hold only
the current shard's ~40k payloads. Peak RAM drops from ~5GB to ~300MB.

The mapping is verified per shard before upserting; if a shard's first/last
id does not line up, that shard falls back to a scan rather than writing
mismatched payloads.

Idempotent: chunk ids are deterministic UUID5s, so re-running upserts the
same points in place instead of duplicating.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHUNKS = ROOT / "chunks" / "chunks.jsonl"
SHARDS = ROOT / "embeddings"
STATE = SHARDS / "_upserted.json"
COLLECTION = os.getenv("QDRANT_COLLECTION", "securegpt_v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
SHARD_SIZE = 40000
BATCH = 1024
PACE_SECONDS = 0.0      # memory now stable on-disk; pacing no longer needed
SHARD_PAUSE = 5.0       # brief settle between shards
DROP_FIELDS = {"embed_text"}


def load_state():
    return set(json.loads(STATE.read_text())) if STATE.exists() else set()


def save_state(done):
    STATE.write_text(json.dumps(sorted(done)))


def upsert_with_retry(client, points, attempts: int = 8):
    """Qdrant closes sockets when its write queue saturates. The write is
    fine -- the connection is what fails -- so back off and let it drain.
    wait=True applies backpressure so we never outrun the server."""
    delay = 2.0
    for i in range(attempts):
        try:
            client.upsert(collection_name=COLLECTION, points=points, wait=True)
            return
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"      retry {i+1}/{attempts} after {type(e).__name__}; {delay:.0f}s",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 1.8, 60)


def payloads_for_lines(fh, start_line, cur_line, count):
    """Advance an open handle to start_line and read `count` payloads."""
    while cur_line < start_line:
        if not fh.readline():
            break
        cur_line += 1
    out = {}
    for _ in range(count):
        line = fh.readline()
        if not line:
            break
        cur_line += 1
        c = json.loads(line)
        for k in DROP_FIELDS:
            c.pop(k, None)
        out[c["chunk_id"]] = c
    return out, cur_line


def iter_point_batches(ids, dense, sp_idx, sp_val, sp_off, payloads, batch):
    """Yield batches of PointStruct instead of materialising all 40,000 first.

    The previous version built every point up front: 40k x 1024 Python floats
    per shard (~2.7GB resident) and many seconds of pure object construction
    before a single byte was sent. Streaming batches keeps memory at ~70MB and
    starts writing immediately, which is where most of the wall-clock time in
    the 568s/shard actually went -- not the network.
    """
    buf, missing = [], 0
    for i, cid in enumerate(ids):
        pl = payloads.get(cid)
        if pl is None:
            missing += 1
            continue
        lo, hi = int(sp_off[i]), int(sp_off[i + 1])
        buf.append(qm.PointStruct(
            id=cid,
            vector={
                "dense": dense[i].tolist(),
                "sparse": qm.SparseVector(
                    indices=sp_idx[lo:hi].astype(int).tolist(),
                    values=sp_val[lo:hi].astype(float).tolist()),
            },
            payload=pl,
        ))
        if len(buf) >= batch:
            yield buf, missing
            buf, missing = [], 0
    if buf or missing:
        yield buf, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-all", action="store_true", help="ignore _upserted.json")
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL, timeout=600, prefer_grpc=True)
    if not client.collection_exists(COLLECTION):
        raise SystemExit(f"{COLLECTION} missing -- run qdrant_setup.py first")

    done = set() if args.force_all else load_state()
    shards = sorted(SHARDS.glob("shard_*.npz"))
    pending = [s for s in shards if s.name not in done]
    print(f"{len(shards)} shards total | {len(done)} already upserted | "
          f"{len(pending)} pending", flush=True)
    if not pending:
        print("Nothing to do.")
        return

    fh = open(CHUNKS, "r", encoding="utf-8")
    cur_line = 0
    t_start = time.time()

    for path in pending:
        k = int(path.stem.split("_")[1])
        z = np.load(path, allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        dense = z["dense"].astype(np.float32)
        sp_idx, sp_val, sp_off = z["sp_indices"], z["sp_values"], z["sp_offsets"]

        start_line = k * SHARD_SIZE
        if start_line < cur_line:          # need to rewind
            fh.close(); fh = open(CHUNKS, "r", encoding="utf-8"); cur_line = 0

        t0 = time.time()
        payloads, cur_line = payloads_for_lines(fh, start_line, cur_line, len(ids))

        if payloads.get(ids[0]) is None or payloads.get(ids[-1]) is None:
            print(f"  {path.name}: line-offset mismatch, falling back to full scan",
                  flush=True)
            want = set(ids)
            payloads = {}
            with open(CHUNKS, "r", encoding="utf-8") as f2:
                for line in f2:
                    c = json.loads(line)
                    if c["chunk_id"] in want:
                        for kk in DROP_FIELDS:
                            c.pop(kk, None)
                        payloads[c["chunk_id"]] = c
            fh.close(); fh = open(CHUNKS, "r", encoding="utf-8"); cur_line = 0

        sent, missing = 0, 0
        for batch_pts, batch_missing in iter_point_batches(
                ids, dense, sp_idx, sp_val, sp_off, payloads, BATCH):
            missing += batch_missing
            if batch_pts:
                upsert_with_retry(client, batch_pts)
                sent += len(batch_pts)
                if sent % (BATCH * 10) == 0:
                    print(f"      {sent:,}/{len(ids):,}", flush=True)

        del payloads, dense, z
        done.add(path.name)
        save_state(done)
        el = time.time() - t0
        note = f" | {missing} missing payloads" if missing else ""
        print(f"  {path.name}: upserted {sent:,} in {el:.0f}s{note}", flush=True)
        # let the optimizer settle between shards
        time.sleep(SHARD_PAUSE)

    fh.close()
    total = client.count(collection_name=COLLECTION, exact=True).count
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. "
          f"{COLLECTION} contains {total:,} points.")


if __name__ == "__main__":
    main()

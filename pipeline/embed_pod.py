#!/usr/bin/env python3
"""
RUNS ON THE RUNPOD GPU POD.

Reads embed_input.jsonl ({"id": ..., "text": ...} per line), embeds every
chunk with BGE-M3 producing BOTH the dense vector and the learned sparse
lexical weights in a SINGLE forward pass, and writes results as shards.

Sharding matters for three reasons:
  * the local machine can download + upsert shard N while the GPU is still
    working on shard N+1
  * a crash costs one shard, not the whole run
  * memory stays bounded regardless of corpus size

Each shard is an .npz holding:
    ids            (N,)      chunk ids, so vectors can be rejoined to the
                             full metadata payload that never left the laptop
    dense          (N, 1024) float16
    sp_indices     (nnz,)    int32   concatenated sparse token ids
    sp_values      (nnz,)    float16 concatenated sparse weights
    sp_offsets     (N+1,)    int64   row boundaries into the two arrays above
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

WORK = Path("/workspace")
IN_FILE = WORK / "embed_input.jsonl.gz"
OUT_DIR = WORK / "shards"
MODEL_NAME = "BAAI/bge-m3"


def load_rows(path: Path):
    """Reads the gzipped id+text file directly -- no decompression step, and
    the upload is 61MB instead of the 2.3GB full chunk file (all the metadata
    stays on the laptop and is rejoined to the vectors at upsert time)."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    ids, texts = [], []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"])
            texts.append(r["text"])
    return ids, texts


def sparse_to_arrays(lexical_weights):
    """BGE-M3 returns [{token_id_str: weight}, ...]. Flatten to CSR-style
    arrays so a shard is a couple of contiguous numpy buffers."""
    idx_parts, val_parts = [], []
    offsets = np.zeros(len(lexical_weights) + 1, dtype=np.int64)
    for i, d in enumerate(lexical_weights):
        if d:
            keys = np.fromiter((int(k) for k in d.keys()), dtype=np.int32, count=len(d))
            vals = np.fromiter((float(v) for v in d.values()), dtype=np.float16, count=len(d))
        else:
            keys = np.empty(0, dtype=np.int32)
            vals = np.empty(0, dtype=np.float16)
        idx_parts.append(keys)
        val_parts.append(vals)
        offsets[i + 1] = offsets[i] + len(keys)
    sp_indices = np.concatenate(idx_parts) if idx_parts else np.empty(0, np.int32)
    sp_values = np.concatenate(val_parts) if val_parts else np.empty(0, np.float16)
    return sp_indices, sp_values, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-size", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    import torch
    from FlagEmbedding import BGEM3FlagModel

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"Loading {MODEL_NAME} (fp16)...", flush=True)
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    ids, texts = load_rows(IN_FILE)
    total = len(ids)
    n_shards = -(-total // args.shard_size)
    print(f"{total:,} chunks -> {n_shards} shards of {args.shard_size:,}", flush=True)

    t_start = time.time()
    done_count = 0

    for s in range(n_shards):
        out_path = OUT_DIR / f"shard_{s:04d}.npz"
        if out_path.exists():
            print(f"[shard {s}] already exists, skipping", flush=True)
            done_count += args.shard_size
            continue

        lo, hi = s * args.shard_size, min((s + 1) * args.shard_size, total)
        shard_ids, shard_texts = ids[lo:hi], texts[lo:hi]
        t0 = time.time()

        out = model.encode(
            shard_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = np.asarray(out["dense_vecs"], dtype=np.float16)
        sp_indices, sp_values, sp_offsets = sparse_to_arrays(out["lexical_weights"])

        tmp = out_path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(shard_ids, dtype=object),
                 dense=dense, sp_indices=sp_indices,
                 sp_values=sp_values, sp_offsets=sp_offsets)
        os.replace(tmp, out_path)   # atomic: syncer never sees a partial shard

        done_count = hi
        el = time.time() - t0
        total_el = time.time() - t_start
        rate = done_count / total_el
        print(f"[shard {s+1}/{n_shards}] {hi-lo:,} chunks in {el:.0f}s "
              f"| {rate:.0f} chunks/s | ETA {(total-done_count)/max(rate,1)/60:.1f} min",
              flush=True)

    (WORK / "EMBED_DONE").write_text("done")
    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

# SecureGPT — Production Chunking, Metadata & Hybrid Embedding Strategy

Status: proposed design, ready to implement. Supersedes the ACPMC chunker in
`SecProj/SecureGPT/backend/quadrant_embeddings/chunking/adaptive_chunking.py`.

---

## 0. Corpus reality (measured, not assumed)

These numbers drive every decision below. Measured with
`scripts/deep_validate_markdown.py` over all 6,641 converted markdown files.

| Metric | Value |
|---|---|
| Total tokens | **81.7M** |
| Documents | 6,641 |
| Markdown tables | 20,378 |
| Est. chunks @ 512 tok | **~178,000** |
| p50 / p90 / p99 doc size | 5.7k / 13.7k / **309k** tokens |
| Largest single doc | 558k tokens |

**Tokens by source type — the decisive fact:**

| Source | Files | Tokens | % of corpus | Avg/file |
|---|---:|---:|---:|---:|
| **.csv** | 694 | **46.7M** | **57%** | 67,349 |
| .pdf | 1,915 | 11.4M | 14% | 5,929 |
| .xlsx | 836 | 11.3M | 14% | 13,532 |
| .txt | 1,704 | 8.1M | 10% | 4,756 |
| .docx | 1,332 | 4.0M | 5% | 3,004 |
| .md | 124 | 0.16M | <1% | 1,254 |
| .pptx | 36 | 0.01M | <1% | 356 |

**Three consequences:**

1. **Tabular data is the corpus, not an edge case.** CSV + XLSX = 71% of all
   tokens. A chunker that treats tables as a special case bolted onto a prose
   splitter will fail on 71% of the data. Tables are the primary path.
2. **Department imbalance is extreme.** Executive_Corporate 32.1M and
   Technology_Digital 29.7M tokens vs Finance 89k. Without metadata filtering,
   any Finance query is competing against 700× more Technology chunks.
   RBAC/department filtering is a *relevance* mechanism here, not only security.
3. **Heavy source-level duplication.** Verified: one boilerplate paragraph
   occurs 96× inside a single source PDF, and the same regional report template
   repeats across Australia/Japan/Singapore/South_Korea. Naive chunking creates
   thousands of near-identical vectors that flood top-k. Needs explicit handling.

---

## 1. Chunking strategy

### Golden rule (unchanged from your design doc — it was right)
> Chunk by document structure first, token count second. Only split by tokens
> when a structural unit exceeds budget.

What's new is **how tables are routed** and **context enrichment**.

### Pipeline

```
markdown file
     ↓
[0] normalize      repair tables, balance fences, strip page artifacts
     ↓
[1] structural parse   → document tree: headings H1–H6, tables, code, lists
     ↓
[2] block routing      → each block goes down a TYPE-SPECIFIC path (never mixed)
     ├── table    → table card + row-group / record chunks
     ├── prose    → heading-scoped recursive split
     ├── slide    → 1 slide + notes = 1 chunk
     └── code     → atomic
     ↓
[3] size normalize    merge < 128 tok siblings, hard-split > 1024 tok
     ↓
[4] context enrich    prepend breadcrumb to the EMBEDDED text
     ↓
[5] link & hash       prev/next ids, parent_block_id, content_hash, near-dup group
     ↓
chunks ready for embedding
```

### Stage 2a — Table routing (this is the part that kept breaking)

**A table never enters the prose splitter. Ever.** The old pipeline ran
pipe-delimited tables through `re.split(r"(?<=[.!?])\s+")`, which fragments rows
on every decimal point and drops the header — that is the root cause of your
table retrieval failures.

Every table produces **a table card plus data chunks**:

**1. Table card (always, 1 per table)** — a semantic summary chunk so that
"which report tracks EBITDA margin by quarter" retrieves the *table*, not a
random slice of rows:

```
Finance > CFO > 01 Financial Planning and Analysis
FIN-FPA-RPT-001_Variance_Analysis_KPI_Dashboard.xlsx
Table: Q1 FY2027 Revenue Variance by Business Unit — Sheet "Business Unit Variance"
Columns (7): Business Unit, Country, Budget Revenue (USD M), Actual Revenue (USD M),
             Variance (USD M), Variance %, Primary Driver
Rows: 7
Sample: Engineering Services — Body & Chassis | India | 84.2 | 88.6 | 4.4 | 5.2% | ...
```

**2. Data chunks — routed by table shape:**

| Table shape | Detection | Chunking | Why |
|---|---|---|---|
| **Fits budget** | ≤ ~600 tok | 1 chunk, whole table | Never split what fits |
| **Entity / registry** | unique ID-ish first col, ≥8 cols, mostly text | **1–4 records per chunk, rendered vertically as `Column: value`** | "Who owns dashboard DSH-0421?" needs *one row*, not a 40-row slice. Vertical rendering keeps every value bound to its column name |
| **Time-series / log** | mostly numeric/date cols, narrow | **Row groups sized by tokens** (not fixed row count), header + separator + caption repeated in every slice | Rows only mean something in aggregate; needs neighbours for trend questions |
| **Very wide row** | single row > budget | 1 record per chunk, vertical | Prevents mid-row splits |

Your corpus makes the entity/registry case matter a lot: the biggest CSVs are
`Enterprise_Dashboard_Catalog` (5,001 rows × 20 cols) and
`BI_Dashboard_and_KPI_Catalog` (5,001 × 17) — every row is an independent
entity with its own ID. Row-group slicing those would be actively harmful.

**Non-negotiables for every table data chunk:**
- header row + separator repeated (so the grid is valid markdown standalone)
- caption + heading path repeated
- `rows X–Y of Z` recorded in metadata *and* rendered in the context header
- `parent_block_id` shared by all slices of the same table → the full table can
  always be reassembled at retrieval time
- **row boundaries are never crossed**

### Stage 2b — Prose routing

Heading-scoped, recursive, structure-respecting:

1. Section (content under one heading) ≤ 512 tok → **one chunk**, done.
2. Else split on paragraph boundaries; keep whole paragraphs together.
3. Paragraph still > 512 tok → split on sentence boundaries.
4. Overlap **64 tokens (~12%)** between prose chunks, snapped to sentence
   boundaries — never mid-sentence.
5. Never split inside a table, code fence, or list.

Note I am **dropping the embedding-similarity merge** from the old ACPMC
chunker. It embedded every sentence just to decide merges — that was the source
of the 48h stall — and it buys very little over heading+paragraph structure when
the documents are already well-structured markdown. Structure is a stronger,
free signal.

### Sizing

| Parameter | Value | Rationale |
|---|---|---|
| Target | **512 tokens** | Dense retrieval precision peaks here; larger chunks dilute the embedding vector |
| Max | 1024 tokens | Hard ceiling before forced split |
| Min | 128 tokens | Below this, merge with sibling |
| Prose overlap | 64 tok (12%) | Preserves cross-boundary context |
| Table overlap | 0 rows | Header repetition replaces overlap |

BGE-M3 accepts 8,192 tokens, but *accepting* long input is not the same as
*embedding it well* — a 4k-token chunk averages into a vague vector that matches
everything weakly. We keep chunks tight for precision and recover context at
retrieval time via neighbour expansion (§5).

### Stage 4 — Context enrichment (biggest cheap win)

Every chunk gets a breadcrumb prepended **to the text that is embedded**:

```
Finance > CFO > 01 Financial Planning and Analysis
FIN-FPA-RPT-001_Variance_Analysis_KPI_Dashboard.xlsx (FY2027 Q1)
Business Unit Variance › Q1 FY2027 Revenue Variance by Business Unit (rows 1–7 of 7)
———
| Business Unit | Country | Budget Revenue (USD M) | ... |
| --- | --- | --- | ... |
| Engineering Services — Powertrain | Germany | 62.5 | ... |
```

Without this, a table slice embeds as meaningless numbers. With it, the chunk
carries its own department, document, section, and time period — which is
exactly what makes retrieval work for orphaned tabular data.

**Store `text` (clean) and `context_header` separately** in the payload: embed
the concatenation, but hand the LLM the clean text plus a properly formatted
citation. This keeps citations exact while embeddings stay context-rich.

### Stage 5 — Duplicate handling

Given the measured 96×-repeated boilerplate: compute `content_hash` (sha256 of
normalized text) per chunk. **Do not delete duplicates** — each belongs to its
own document and is needed for correct citation. Instead:
- store `content_hash` in the payload
- **collapse by hash at retrieval time**, keeping the highest-ranked instance

This preserves citation integrity while stopping one boilerplate paragraph from
occupying all ten top-k slots.

---

## 2. Metadata schema

Designed so that (a) RBAC is enforceable server-side, (b) time/department
filtering is exact rather than string-matched, and (c) context can be
reconstructed with no loss.

```jsonc
{
  // ---- identity & provenance ----
  "chunk_id":        "uuid5(doc_id + block_id + chunk_index)",  // deterministic → idempotent re-ingest
  "doc_id":          "uuid5(relative_source_path)",
  "source_path":     "Finance/CFO/01 Financial Planning and Analysis/FIN-FPA-RPT-001_....xlsx",
  "file_name":       "FIN-FPA-RPT-001_Variance_Analysis_KPI_Dashboard.xlsx",
  "source_format":   "xlsx",
  "doc_title":       "FP&A Variance Analysis & KPI Dashboard",
  "content_hash":    "sha256[:16]",          // near-dup collapsing
  "ingested_at":     "2026-08-28T12:00:00Z",
  "pipeline_version":"2.0.0",

  // ---- ACCESS CONTROL (always filtered server-side) ----
  "department":      "Finance",              // keyword index, is_tenant=true
  "sub_department":  "CFO",                  // keyword
  "access_roles":    ["CFO", "CEO"],         // KEYWORD index — never TEXT (see §6)
  "sensitivity":     "internal",             // public | internal | confidential | restricted

  // ---- navigation / structure ----
  "functional_area": "01 Financial Planning and Analysis",
  "heading_path":    ["Business Unit Variance", "Q1 FY2027 Revenue Variance by Business Unit"],
  "section_title":   "Q1 FY2027 Revenue Variance by Business Unit",
  "chunk_type":      "table_rows",           // prose | table_card | table_rows | table_record
                                             // | slide | code | list
  "chunk_index":     12,
  "total_chunks":    47,

  // ---- citation locators (per your citation framework) ----
  "page":            24,                     // pdf
  "sheet":           "Business Unit Variance",// xlsx
  "slide":           null,                   // pptx
  "row_range":       [1, 7],                 // xlsx/csv
  "row_total":       7,

  // ---- CONTEXT RECOVERY (no data loss) ----
  "prev_chunk_id":   "uuid…",                // neighbour expansion
  "next_chunk_id":   "uuid…",
  "parent_block_id": "uuid…",                // all slices of one table share this
  "parent_doc_id":   "uuid…",

  // ---- temporal (parsed, not string-matched) ----
  "fiscal_year":     2027,                   // integer → range queries work
  "fiscal_quarter":  1,
  "calendar_year":   2026,
  "doc_date":        "2026-07-17",

  // ---- retrieval aids ----
  "token_count":     498,
  "has_table":       true,
  "table_columns":   ["Business Unit","Country","Budget Revenue (USD M)", "..."],
  "identifiers":     ["FIN-FPA-RPT-001", "FIN-FPA-POL-001"],  // exact-match lookup

  // ---- content ----
  "text":            "clean chunk text for the LLM + citation",
  "context_header":  "Finance > CFO > … › rows 1–7 of 7"
}
```

**Why each block earns its place**

- `access_roles` as **keyword** (not text) is the difference between working
  RBAC and a security hole (§6).
- `fiscal_year`/`fiscal_quarter` as **integers** replaces the current
  `MatchText(file_name, "2027")` hack — which also matches `Q3_2027_v2.docx`,
  `20271`, and any doc that merely mentions 2027.
- `parent_block_id` + `prev/next_chunk_id` are the no-data-loss mechanism (§5).
- `identifiers` gives exact-match retrieval for document codes
  (`FIN-FPA-RPT-001`) that dense vectors handle poorly.
- `table_columns` enables "which tables have an EBITDA column" style filtering.

---

## 3. Embedding strategy

### Dense: BGE-M3 — confirmed, keep it
1024-dim, cosine, multilingual, 8,192 ctx, strong on BEIR/MIRACL. Correct choice.

### Sparse: your question — BM25, or something else?

**Recommendation: BGE-M3's own learned sparse (lexical weights).**

The decisive reason: **BGE-M3 is a multi-vector model — one forward pass emits
dense *and* sparse (and ColBERT) together.** You get the sparse vector for
essentially free, on the same GPU pass, same tokenizer, same model you're
already loading:

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
out = model.encode(texts, return_dense=True, return_sparse=True, batch_size=32)
# out["dense_vecs"], out["lexical_weights"]
```

Comparison:

| Option | Quality | Cost | Verdict |
|---|---|---|---|
| **BGE-M3 sparse** | Learned weights; some term expansion; strong with dense | **Free** — same forward pass | ✅ **Use this** |
| BM25 | Pure lexical; excellent for exact IDs; no semantics | Cheap (CPU) but needs corpus IDF, separate index | Optional 3rd signal |
| SPLADE | Best-in-class learned sparse | **2× embedding time** — a whole second model + GPU pass | ✗ not worth 2× for a marginal gain |

**The one thing BGE-M3 sparse is weaker at** is exact rare-token matching —
codes like `FIN-FPA-RPT-001` or `CP-00394`, which your corpus is full of. Rather
than add BM25 for this, cover it with a **Qdrant full-text payload index** on
`text` / `file_name` / `identifiers`. That's a keyword path with zero extra
model cost, and it's exact rather than probabilistic.

So the final retrieval signal set is **three paths, one model**:
dense (semantic) + BGE-M3 sparse (lexical) + full-text index (exact IDs).

**Skip ColBERT multi-vector.** BGE-M3 emits it, and Qdrant supports it, but it
stores one vector *per token* — at ~178k chunks that's an enormous index for a
gain the cross-encoder reranker delivers more cheaply.

### Reranking: BGE-reranker-v2-m3 (the single biggest quality lever)

Hybrid fusion gets the right chunk into the top ~50. A cross-encoder gets it
into the top 3. Retrieve 50–100 fused → rerank → pass 5–8 to the LLM.
Same family as BGE-M3, so tokenization and multilinguality stay consistent.

---

## 4. Qdrant collection design

Single collection — your "one vector DB" decision was right (no duplicate
embeddings, central index, security via filters).

```python
client.create_collection(
    collection_name="securegpt_v2",
    vectors_config={
        "dense": VectorParams(
            size=1024, distance=Distance.COSINE,
            quantization_config=ScalarQuantization(              # 4× less RAM
                scalar=ScalarQuantizationConfig(type="int8", always_ram=True)
            ),
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
    },
)
```

**Payload indexes** (this is what makes filtering fast *and* correct):

| Field | Index type | Purpose |
|---|---|---|
| `department` | keyword, `is_tenant=True` | primary partition — Qdrant co-locates tenant data on disk |
| `access_roles` | **keyword** | RBAC — exact match only |
| `sub_department`, `chunk_type`, `source_format` | keyword | filtering |
| `fiscal_year`, `fiscal_quarter`, `token_count` | integer | range queries |
| `doc_id`, `parent_block_id`, `content_hash` | keyword | context recovery, dedup |
| `text`, `file_name`, `identifiers` | **full-text** | exact ID / phrase lookup |

Storage estimate at ~178k chunks: dense int8-quantized ≈ **180 MB**
(vs 730 MB unquantized), plus sparse and payload. Comfortable.

---

## 5. Retrieval pipeline — and how "no data loss" is guaranteed

```
query
  ↓ authenticate → role
  ↓ query analysis: extract dept / fiscal year / doc IDs → build Filter
  ↓ MANDATORY filter: access_roles MATCH_ANY user_roles      ← security boundary
  ↓ Qdrant Query API, single round-trip:
      prefetch[ dense(limit 100), sparse(limit 100) ] → FusionQuery(RRF)
  ↓ collapse duplicates by content_hash
  ↓ BGE-reranker-v2-m3 → top 5–8
  ↓ CONTEXT EXPANSION  ← the no-data-loss step
  ↓ LLM + citations
```

**Five mechanisms that prevent losing context:**

1. **Context header on every chunk** — chunk is self-describing even in isolation.
2. **Neighbour expansion** — after reranking, fetch `prev_chunk_id`/`next_chunk_id`
   for the top hits and stitch them in. A boundary-split answer stays complete.
3. **Table reassembly** — if a hit is `table_rows`, fetch all chunks sharing
   `parent_block_id` to hand the LLM the whole table when it fits.
4. **Header repetition** — every table slice is a valid standalone table.
5. **Small-to-big** — embed precise 512-token chunks for matching accuracy, but
   optionally return the parent section for generation.

Filter-then-search (your doc's instinct) is exactly right, and Qdrant does this
correctly — its filterable HNSW applies the filter *during* graph traversal, so
you don't lose recall the way naive post-filtering does.

---

## 6. Bugs found in the existing SecureGPT code

Worth fixing regardless of which chunker you adopt:

1. **🔴 RBAC index type is wrong — security-relevant.**
   `qdrant_store.py` creates `access_roles` (and `department`) with
   `PayloadSchemaType.TEXT`. Text indexes are *tokenized substring* matches, not
   exact. Combined with `MatchValue`, behaviour on an array field indexed as
   text is not the exact set-membership check RBAC needs. **Must be `KEYWORD`.**
2. **🟠 `ingest.py` is broken at import.** It imports
   `retrieval_pipeline.retrieval.chunking.adaptive_chunking` and
   `...embedding.bge_embedding`, neither of which exists at that path.
3. **🟠 Year/quarter filtering is string-matching filenames.**
   `MatchText(key="file_name", text="2027")` also matches `20271`, `FY2027_v2`,
   or any filename containing the digits. Use parsed integer fields.
4. **🟡 Chunk IDs collide across departments.**
   `make_chunk_id(department, file_name, chunk_index)` uses only the *file name*,
   so two files with the same name in different folders of the same department
   overwrite each other. Use the full relative path.
5. **🟡 Table text is not valid markdown.** Extractors emit `" | ".join(cells)`
   with no separator row, so nothing downstream can detect a table. Fixed in the
   new converters.

---

## 7. Execution plan

| Step | What | Where | Est. |
|---|---|---|---|
| 1 | Implement chunker + metadata builder | local | — |
| 2 | Chunk full corpus, write `chunks.jsonl` | local CPU, parallel | ~10 min |
| 3 | **Review chunk-quality report** (samples per type, size histogram) | — | checkpoint |
| 4 | Embed dense+sparse with BGE-M3 | **RunPod GPU** | ~1–3 h |
| 5 | Create collection + indexes, upsert | RunPod → Qdrant | ~30 min |
| 6 | Retrieval smoke tests, esp. table questions | local | checkpoint |

Chunking runs locally — it's pure CPU string work, no model, minutes not hours.
Only step 4 needs the GPU.

---

## 8. Evidence base — techniques evaluated (researched Aug 2026)

Decisions below are backed by published benchmarks, not preference.

### ✅ ADOPT — Contextual Retrieval (Anthropic)
Prepend situating context to each chunk *before* embedding.

| Configuration | Retrieval failure rate (top-20) | Reduction |
|---|---|---|
| Baseline | 5.7% | — |
| + Contextual embeddings | 3.7% | **35%** |
| + Contextual BM25/sparse | 2.9% | **49%** |
| + Reranking | 1.9% | **67%** |

Cost with prompt caching: **~$1.02 per million document tokens** → ~$83 for
our full 81.7M-token corpus, or **~$25 if scoped to prose only** (PDF+DOCX+TXT
= 23.5M tokens).

**Two-tier implementation:**
- **Tier 1 — deterministic breadcrumb (free, instant, all chunks).** Department
  › document › heading path › row range. For our corpus this captures most of
  the value, because 71% of tokens are tables whose context *is* structural.
- **Tier 2 — LLM-generated context (prose chunks only, optional).** A 50–100
  token "situate this chunk in the document" line. Worth it for narrative
  PDF/DOCX where structure alone doesn't convey meaning. Skip for tables — an
  LLM summary of a row of numbers adds little over the breadcrumb.

### ❌ REJECT — Late chunking (Jina)
Embed the whole document, *then* mean-pool token embeddings per chunk. Elegant,
and genuinely better on multi-paragraph prose — but it **requires mean pooling**.
BGE-M3's dense vector is the **normalized `[CLS]` token** hidden state
(`pooling_method='cls'` in FlagEmbedding). Mean-pooling BGE outputs departs from
how the model was trained and degrades quality, and query embeddings would no
longer live in the same space as document embeddings. Adopting late chunking
would mean switching to a mean-pooling long-context model (e.g.
jina-embeddings-v3) — a much larger change than it appears. **Not with BGE-M3.**

### ❌ REJECT — Semantic / embedding-similarity chunking
This is what the old ACPMC chunker did (embed every sentence, merge by cosine
similarity ≥ 0.75) and it is what caused the 48-hour stall. The evidence says it
also wasn't buying quality:

- NAACL 2025 (*"Is Semantic Chunking Worth the Computational Cost?"*): costs
  **are not justified by consistent performance gains**.
- Feb 2026 benchmark, 7 strategies over 50 real documents: **recursive
  512-token splitting ranked #1 at 69% accuracy; semantic chunking 54%**.

Structure-first chunking on well-formed markdown is both cheaper and better.

### ✅ CONFIRM — 512-token target
Recursive splitting at **512 tokens with 50–100 token overlap** is the
benchmark-validated default (69% accuracy, largest 2026 real-document test).
Query-type dependency: factoid queries favour 256–512, analytical favour 1024+.
512 with neighbour expansion at retrieval time serves both.

Note this guidance is about **prose**. For entity records the record *is* the
semantic unit — a complete 224-token dashboard record beats 512 tokens
containing 2.3 unrelated records.

### ⏸ DEFERRED — Cross-encoder reranking
Biggest remaining lever (49% → 67% failure reduction). Deferred by decision;
the metadata and retrieval design leave a clean slot for it.

### ❌ NOT NEEDED — RAPTOR / GraphRAG
Recursive summarization trees and knowledge graphs pay off on densely
cross-referential narrative corpora. Ours is 71% tabular records with strong
explicit structure already captured in metadata. High cost, low marginal gain.

---

## 9. The document-level path (no-data-loss escape hatch)

For questions where chunk retrieval is structurally the wrong tool — "teach me
this 500-page document", "summarise this entire report" — the answer is not
bigger chunks. It is **retrieving the document, not chunks**, and handing the
whole thing to a long-context model. This is a mainstream 2026 pattern (agentic
RAG: the agent decides *what* to retrieve and at what granularity).

**We get this almost for free, because `markdown/` already is the document store.**
No re-parsing is needed — the converted markdown is on disk, keyed by the same
`doc_id` stored in every chunk payload:

```
query → hybrid search (chunks)
      → inspect top hits' doc_id / frequency
      → if the question is document-scoped:
            load markdown/<source_path>.md   ← already converted, instant
            feed whole document to long-context LLM
      → else: normal chunk-level answer
```

Three routes out of one index:

| Question shape | Route | Mechanism |
|---|---|---|
| "Who owns dashboard DSH-0421?" | chunk retrieval | hybrid + filter |
| "Summarise the FY2026 APAC report" | **document load** | `doc_id` → read `.md` from disk |
| "Total revenue across all business units" | **structured/agent** | metadata filter + scroll, or coding agent over the CSV |

This is why the metadata schema carries `doc_id`, `source_path` and
`parent_block_id` — they are the handoff points between the three routes.

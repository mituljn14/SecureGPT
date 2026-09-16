# CSV Calculation Benchmark

Ten ground-truthed calculation questions for the coding agent, each over a
**different CSV file** in a **different functional area** of the corpus.

## Why these questions

Every target file holds 600–5,000 rows. That is deliberate: the answer cannot
be read off the schema preview the model receives in its prompt (3 rows). To
answer, the agent must retrieve the right file, mount it in the Docker
sandbox, and run real pandas code over every row. A model that guesses,
pattern-matches, or computes from the preview gets it wrong.

Expected values are **recomputed from the CSVs at run time** by
`recompute_ground_truth()` — they are never hard-coded, so they cannot drift
out of sync if the corpus is rebuilt.

| # | Area | File | Calculation |
|---|------|------|-------------|
| 1 | Operations / COO / Logistics | `Shipment_Data_Q1_2017.csv` | `sum(Cost_INR)` |
| 2 | Operations / COO / Production | `Work_Order_Data_Q1_2017.csv` | `sum(Scrap_Qty)` |
| 3 | Technology / CTO / Cybersecurity | `Vulnerability_Scan_Log_..._Q1_2023.csv` | `mean(CVSS_Score)` |
| 4 | Technology / CTO / Machine Learning | `Experiment_Log_..._Q1_2018.csv` | `mean(Val_Accuracy)` where `Status == Completed` |
| 5 | Operations / CEngO / Electric Mobility | `EV_Efficiency_BattPack_Q1_2017.csv` | `mean(Efficiency_Pct)` |
| 6 | Operations / CEngO / Connected Vehicles | `CV_Events_CloudConn_Q1_2018.csv` | `mean(Latency_ms)` |
| 7 | Executive / Regional MD / India | `Bengaluru_Regional_Operations_Registry_Part_1.csv` | `sum(Revenue_INR)` |
| 8 | Technology / CDO / AI Models | `Enterprise_AI_Model_Registry_Part_1.csv` | `mean(F1_Score)` |
| 9 | Legal / CLO / Contracts | `14_Contract_Obligation_Tracker.csv` | `sum(Contract_Value)` |
| 10 | Technology / CTO / AI Models | `Training_Runs_..._Q2_2019.csv` | `min(Val_Loss)` |

Case 4 is the only one with a filter clause — it checks the agent applies a
`WHERE`-style condition rather than aggregating the whole column (257 of 753
rows qualify).

## Running it

Everything below assumes `backend/` as the working directory and the project
venv. Qdrant and Ollama must be up, and the `coding-agent-sandbox` Docker
image must exist.

Fast pre-flight — does retrieval mount the right file? No LLM, no Docker,
about a minute:

```bash
cd backend && ./venv/Scripts/python.exe -u -m evaluation.csv_calc_benchmark --retrieval-only
```

Full end-to-end, all ten:

```bash
cd backend && ./venv/Scripts/python.exe -u -m evaluation.csv_calc_benchmark --json ../benchmark_results.json
```

A single case, for a live demo:

```bash
cd backend && ./venv/Scripts/python.exe -u -m evaluation.csv_calc_benchmark --only 7
```

`-u` matters: without it Python buffers stdout and you see nothing until the
whole run finishes.

Other flags: `--role` (default `CEO`, which can see the whole corpus — use a
narrower role to demonstrate RBAC filtering), `--json PATH` to write
machine-readable results.

Exit code is 0 only if every selected case passes.

## How an answer is judged

`coerce_number()` pulls a single number out of whatever the agent returned and
compares it to ground truth within 1% relative tolerance. It handles the forms
this corpus actually produces: thousands separators, a leading `₹`/`$`,
scientific notation (`1.2454e+12`), and Indian magnitude words (`crore`,
`lakh`) as multipliers. A non-numeric answer — including an honest
`NOT_FOUND_IN_PROVIDED_DATA` — counts as a failure, because every one of these
questions is answerable from the corpus.

The harness also records, per case, whether the correct file was mounted at
all (`right_file`). That separates a **retrieval** failure from a **code
generation** failure, which are fixed in completely different places.

## The failure modes this benchmark exists to catch

Retrieval was never the hard part. Once file selection was fixed (correct file
ranked first for 10/10), every remaining failure was the code-generation model
failing to ground itself in information it had already been given. Four
distinct shapes, in descending order of danger:

1. **Phantom period filter.** The question says "Q1 2017", so the model filters
   for it — against a file that *is* Q1 2017. `df[df['Date'].str.contains('Q1-2017')]`
   matches zero rows (the dates are ISO, `2017-03-19`), and the sum comes back
   `0.0`. No exception, a valid number, completely wrong. Countered by the
   `PERIOD SCOPE:` line in the prompt.
2. **Hand-rolled CSV parsing.** After a retry, the model abandoned pandas for
   `open()`/`split(',')` and took `parts[1]` as the cost column — position 1 is
   `Origin`; the cost is `Cost_INR` at position 6. It crashed on
   `float('Origin')`, which was luck: any numeric column in that position would
   have produced a confident total of the wrong field. Caused by an unscoped
   "avoid brittle string splits" prompt rule meant for text documents; now
   scoped, with an explicit "CSV/XLSX always use pandas, address by name".
3. **Invented column names.** `Total_Cost_INR` / `Shipment_Cost_INR` derived
   from the question's wording rather than the schema. Caught statically by
   `validate_column_references()` before any container starts.
4. **Invented filter columns.** `df['Year'] == 2017`, `df['Type'] == 'Logistics'`.
   Same static guard catches these, and says to remove the filter rather than
   rename it.

Only (3) and (4) raise `KeyError` and are therefore self-announcing. (1) and (2)
return plausible numbers, which is why the benchmark asserts against
independently computed ground truth rather than merely checking that a number
came back.

## Comparing code-generation models

`--model` overrides the Ollama model for code generation, leaving retrieval and
everything else identical:

```bash
cd backend && ./venv/Scripts/python.exe -u -m evaluation.csv_calc_benchmark --only 1,2 --model ministral-3:8b
```

This matters because failure modes (1)–(4) are all model-capability limits, not
pipeline bugs. The guards convert them from wrong answers into retries, but a
model that needs three corrected attempts per question is not a comfortable
thing to demo. Run the same cases across models before settling the config.

## Known limitations

**Q2 names its column.** The question says `Scrap_Qty` explicitly. Without
that token, hybrid search ranks `COO_*_Catalog.{csv,xlsx}` above the actual
data file — a catalog that *lists* every production report is textually very
similar to a question *about* production reports, and only the sparse/BM25
half of the query separates them. The other nine questions need no such hint.
A general fix would demote index/catalog documents for calculation queries;
that is not implemented.

**Role is `CEO`.** CEO has access to every chunk in the corpus, so this
benchmark does not exercise RBAC narrowing. Pass `--role` to test that
separately.

**Latency is the local model.** Each question costs one or more
`qwen2.5-coder:3b` calls on CPU plus a Docker run. Expect roughly 1–2 minutes
per question, dominated by prompt evaluation, and longer on the first question
while BGE-M3 loads.

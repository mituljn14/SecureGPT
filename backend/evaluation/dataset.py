"""
dataset.py

The golden query set for evaluation/ragas_eval.py.

This is deliberately a small, hand-curated, EXTENSIBLE seed set, not an
attempt to enumerate "the hundreds of queries people can ask" -- that
corpus-spanning coverage is exactly what code-level fixes (never a fixed
query list) have to handle. What this set is for: a regression net that
would have caught every real bug found and fixed this session, so future
changes to the pipeline get checked against them automatically instead of
by hand, one screenshot at a time.

Add rows freely -- each is independent. Every row needs:
    query               the user's question, verbatim
    role                a role from api/database.py MOCK_USERS
    expect_destination  "__direct__" | "enterprise" | "coding" -- what
                         supervisor_node SHOULD route this to. This alone
                         catches the two biggest bugs found this session:
                         the RRF-confidence-always-1.0 bug (never
                         escalated to agentic retrieval) and the
                         direct-answer hallucination-bypass bug (a company
                         question answered with zero grounding).

Optional, depending on what the row is testing:
    ground_truth            a verified reference answer/fact, used for
                             ragas context_recall / context_precision.
                             Leave None when you don't have an
                             independently-verified value -- faithfulness
                             and answer_relevancy (used for every row)
                             don't need one, and a made-up "ground truth"
                             would just be a second place to be wrong.
    expect_answer_contains   substring(s) the coding agent's numeric
                             answer must contain (e.g. "5956.2"). Only
                             meaningful for expect_destination="coding" --
                             ragas' context-based metrics don't apply
                             there (see ragas_eval.py's module docstring
                             for why).
    expect_no_fabrication    True marks a query that must NOT get a
                             confident answer at all -- either the corpus
                             genuinely has nothing (nonsense/out-of-scope
                             question) or the row is testing RBAC (a role
                             asking about another department's data it
                             shouldn't see). The check: the final answer
                             must not read as a confident, specific
                             factual claim while retrieved_chunks came
                             back empty (or, for RBAC rows, must not
                             surface the other department's specifics).
    expect_generated_file    True means a downloadable file must come out
                             the other end -- catches the file_writer.py
                             OUTPUT_DIR / node_write_output_file bug found
                             this session (the file was written but never
                             reachable via any download link).
    note                 why this row exists / what bug it guards against.
"""

DATASET = [
    # ---- Tier 1: instant greeting fast-path (0 LLM calls) -----------------
    {
        "query": "hey",
        "role": "CEO",
        "expect_destination": "__direct__",
        "note": "Greeting dict fast-path -- must never reach the LLM or Qdrant.",
    },
    {
        "query": "thank you",
        "role": "CFO",
        "expect_destination": "__direct__",
        "note": "Greeting dict fast-path, different role.",
    },

    # ---- Tier 2: casual chat / general knowledge, NOT in the dict ---------
    {
        "query": "How's it going today?",
        "role": "CEO",
        "expect_destination": "__direct__",
        "expect_no_fabrication": True,
        "note": "Casual chat outside the greeting dict -- must go through the "
                "LLM classify-and-answer call, and the corpus safety-net probe "
                "must NOT misfire and reroute small talk into enterprise RAG.",
    },
    {
        "query": "What is Tesla's main product line?",
        "role": "CEO",
        "expect_destination": "__direct__",
        "expect_no_fabrication": True,
        "note": "Another company, general knowledge -- must answer directly "
                "about Tesla, not blend in fabricated Nova/NAES data.",
    },

    # ---- Supervisor hallucination-bypass safety net (the fix from this session) ----
    {
        "query": "What did the risk assessment report flag as our top concerns?",
        "role": "CLO",
        "expect_destination": "enterprise",
        "note": "Doesn't literally say 'Nova' or 'our company' -- exactly the "
                "phrasing gap that let the single-call supervisor classify a "
                "real company question as general knowledge and answer it "
                "ungrounded. Tests the corpus-relevance override in "
                "supervisor.py (_corpus_has_real_match).",
    },
    {
        "query": "Give me a rundown of the latest procurement performance numbers.",
        "role": "CFO",
        "expect_destination": "enterprise",
        "note": "Same override test, Finance-flavored phrasing.",
    },

    # ---- Enterprise RAG: single-year, multi-year range, cross-department ----
    {
        "query": "What were the key highlights from the 2022 annual report?",
        "role": "CEO",
        "expect_destination": "enterprise",
        "note": "Single-year fact retrieval, broadest-access role.",
    },
    {
        "query": "Summarize revenue trends from 2014 to 2018.",
        "role": "CFO",
        "expect_destination": "enterprise",
        "note": "Year-RANGE parsing (query_metadata_parser.py) -- must not "
                "collapse to fiscal_year==2014 only.",
    },
    {
        "query": "What did the board discuss in the most recent meeting minutes?",
        "role": "Corporate_Strategy",
        "expect_destination": "enterprise",
        "note": "Executive/board content, non-Finance role.",
    },
    {
        "query": "What is our current compliance audit status?",
        "role": "CLO",
        "expect_destination": "enterprise",
        "note": "Legal/Compliance department query.",
    },

    # ---- Coding agent: the actual hallucination bug this session started from ----
    {
        "query": "calculate net cash from operations from 2018-2022",
        "role": "CEO",
        "expect_destination": "coding",
        "expect_answer_contains": ["5956.2", "5,956.2"],
        "note": "THE bug: previously fabricated [1000,1200,1500,1800,2000] -> "
                "7500. Verified ground truth from Annual_Report_2022.pdf: "
                "699.3+1092.5+1629.8+966.8+1567.8 = 5956.2. Role MUST be one "
                "with real access to Annual_Reports (CEO/Corporate_Strategy "
                "only, confirmed live) or this tests RBAC-decline instead --"
                " see the CFO row below for that case.",
    },
    {
        "query": "calculate net cash from operations from 2018-2022",
        "role": "CFO",
        "expect_destination": "coding",
        "expect_no_fabrication": True,
        "note": "SAME query, a role whose RBAC scope does NOT include "
                "Annual_Reports (found live: access_roles on that document "
                "is ['Corporate_Strategy', 'CEO'] only). Retrieval correctly "
                "returns other Finance chunks that don't actually contain "
                "this figure (e.g. a Treasury Operations policy manual) -- "
                "must decline honestly (llm_agent.py's NOT_FOUND_IN_PROVIDED_DATA "
                "contract), not answer a plausible-looking wrong number like "
                "the '1000' seen live before this fix.",
    },
    {
        "query": "What is the average of 45, 67, and 89?",
        "role": "CEO",
        "expect_destination": "coding",
        "expect_answer_contains": ["67"],
        "note": "Genuine standalone calculation with no corpus data needed -- "
                "must NOT be dragged into enterprise retrieval by the "
                "always-retrieve-first change, and must not be blocked by "
                "the relevance gate (is_standalone_query fallback).",
    },
    {
        "query": "Calculate total revenue from 2014 to 2018 and save it as a csv report",
        "role": "CEO",
        "expect_destination": "coding",
        "expect_generated_file": True,
        "note": "File-export path (CSV) -- catches the file_writer.py OUTPUT_DIR "
                "bug (files written to a folder the /download route never served "
                "from, and never even attached to generated_files). Role must "
                "have real access to Annual_Reports (same scoping as the net "
                "cash rows above) or this tests RBAC-decline instead of export.",
    },
    {
        "query": "Calculate total revenue from 2014 to 2018 and save it as a pdf report",
        "role": "Corporate_Strategy",
        "expect_destination": "coding",
        "expect_generated_file": True,
        "note": "Same file-export path, PDF specifically -- this format was "
                "entirely missing before this session's fix: not in "
                "file_writer.py's ALLOWED_EXTENSIONS, and "
                "detect_output_file_request() had no 'as pdf' branch at all, "
                "so a PDF export request silently produced no file and no "
                "error ('if i am unable to download pdf').",
    },

    # ---- RBAC: role must not see another department's specifics ----------
    {
        "query": "What is the exact salary budget breakdown by employee for the Finance department?",
        "role": "CHRO",
        "expect_destination": "coding",
        "expect_no_fabrication": True,
        "note": "Sensitive/cross-department financial detail requested by a "
                "role whose access_roles scope may not include it -- answer "
                "must come from whatever this role's RBAC-filtered retrieval "
                "actually returns, never a fabricated-sounding specific "
                "figure standing in for data the role can't see.",
    },

    # ---- Out-of-corpus: must decline, not invent ---------------------------
    {
        "query": "What is our company's detailed plan for colonizing Mars?",
        "role": "CEO",
        # __direct__, not enterprise: the supervisor's corpus probe finds no
        # match above the relevance floor, so declining directly is the
        # CORRECT outcome. The earlier "enterprise" expectation predated that
        # safety net and mis-scored a right answer as a routing failure.
        "expect_destination": "__direct__",
        "expect_no_fabrication": True,
        "note": "Nonsense-but-company-phrased query with certainly nothing "
                "in the corpus -- must come back as 'not found', not a "
                "confidently invented plan.",
    },

    # ---- Verified ground truth: unlocks ContextPrecision / ContextRecall ---
    # Values checked directly against corpus/Operations/COO/Production_Reports/
    # Work_Order_Data_Q3_2022.csv (awk over the source file), not from any
    # model output. Without a reference answer the retrieval-quality metrics
    # cannot run at all.
    {
        "query": "What was the scrap quantity for work order WO-202207-000550?",
        "role": "COO",
        "expect_destination": "enterprise",
        "ground_truth": "Work order WO-202207-000550 (Forged Crankshaft, Chakan "
                        "Facility, Body Shop Line 2) had a scrap quantity of 13.",
        "note": "Single-value lookup from Work_Order_Data_Q3_2022.csv; "
                "Scrap_Qty column, verified = 13. Exact-identifier query: "
                "the first 10-row GPU run scored context recall 0.0 here "
                "because dense/sparse similarity never surfaced the chunk; "
                "retrieval/identifier_lookup.py was added in response.",
    },
    {
        "query": "Which plant and line handled work order WO-202207-000342?",
        "role": "COO",
        "expect_destination": "enterprise",
        "ground_truth": "Work order WO-202207-000342 was handled at the Chennai "
                        "Plant on Trim & Final Assembly Line 2 (Suspension Arm, "
                        "scrap quantity 22).",
        "note": "Same source file, different row; plant/line/scrap verified "
                "against the CSV. The earlier ground truth did not name the "
                "plant, so a correct answer scored context recall 0.0.",
    },

    # ---- RBAC isolation: same question, two roles, opposite outcomes -------
    # This is the check we had been doing by hand. Operations data must be
    # readable by COO and invisible to CHRO; the CHRO turn must decline
    # rather than answer from documents outside its scope.
    {
        "query": "What was the scrap quantity for work orders in Q3 2022?",
        "role": "COO",
        "expect_destination": "enterprise",
        "note": "RBAC pair (allowed): COO owns Operations documents.",
    },
    {
        "query": "What was the scrap quantity for work orders in Q3 2022?",
        "role": "CHRO",
        "expect_destination": "enterprise",
        "expect_no_fabrication": True,
        "note": "RBAC pair (denied): CHRO cannot read Operations. Retrieval "
                "returns only People_Culture chunks, so the answer must be an "
                "honest 'not found' -- never a scrap figure.",
    },

    # ---- Multimodal routing: agents that postdate the original dataset ----
    # These exercise routing only (an attachment decides the destination
    # before any LLM runs). The uploaded_files paths do not need to exist
    # for the supervisor's decision; the agent nodes are not what's under
    # test here.
    {
        "query": "Generate the minutes for this recording",
        "role": "CEO",
        "expect_destination": "meeting",
        "uploaded_files": ["standup.mp3"],
        "routing_only": True,
        "note": "Audio attachment -> meeting agent, no LLM call.",
    },
    {
        "query": "Generate the minutes for this recording",
        "role": "CEO",
        "expect_destination": "meeting",
        "uploaded_files": ["standup.mp3 (1).mpeg"],
        "routing_only": True,
        "note": "The re-download rename that broke routing before: .mpeg "
                "must still reach the meeting agent.",
    },
    {
        "query": "What does this diagram show?",
        "role": "CEO",
        "expect_destination": "vision",
        "uploaded_files": ["schematic.png"],
        "routing_only": True,
        "note": "Image attachment -> vision agent.",
    },
    {
        "query": "Visualise this data and generate a bar chart",
        "role": "CEO",
        "expect_destination": "vision",
        "uploaded_files": ["sales.csv"],
        "routing_only": True,
        "note": "Spreadsheet + chart intent -> vision (ChartGenerator), "
                "NOT coding.",
    },
    {
        "query": "Calculate the total revenue in this file",
        "role": "CEO",
        "expect_destination": "coding",
        "uploaded_files": ["sales.csv"],
        "routing_only": True,
        "note": "Same spreadsheet, computation intent -> coding. The "
                "intent boundary, not the file type, decides.",
    },
]

# Data Scope Decisions
Generated from `data.zip` (7982 files walked).

## IN SCOPE — copied to `corpus/` (6815 files)
| Extension | Count | Reason |
|---|---|---|
| .pdf | 1978 | core business document |
| .txt | 1710 | plain text document |
| .docx | 1392 | core business document |
| .xlsx | 860 | core business document (tabular) |
| .csv | 702 | core business document (tabular) |
| .md | 130 | plain markdown document |
| .pptx | 43 | core business document (slides) |

## OUT OF SCOPE — deferred, left in `raw/` only (1150 files)
| Extension | Count | Reason |
|---|---|---|
| .json | 358 | structured config/data — ambiguous (could be real business data); deferred pending your review |
| .py | 240 | source code — not organizational knowledge prose; revisit if CTO/CIO runbooks need indexing |
| .yaml | 173 | structured config/data — same ambiguity as .json; deferred pending your review |
| .sql | 141 | DB schema / query scripts — same reasoning as source code |
| .svg | 130 | vector image/diagram — same as .png |
| .msg | 38 | Outlook email — needs a dedicated .msg parser (extract-msg); deferred, not skipped forever |
| .png | 28 | image/diagram — needs OCR or a vision captioner to be searchable; deferred |
| .xml | 26 | technical config, not business prose/tabular knowledge; deferred |
| .cpp | 8 | source code — same as .py |
| .java | 6 | source code — same as .py |
| .yml | 1 | structured config/data — same ambiguity as .json; deferred pending your review |
| .js | 1 | source code — same as .py |

## UNCLASSIFIED — needs a decision (17 files)
| Extension | Count |
|---|---|
| (no extension) | 14 |
| .ps1 | 2 |
| .tf | 1 |

Full per-file breakdown: `docs/manifest.csv`

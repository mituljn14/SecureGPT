# Data Quality Issues Found During Scoping

## 1. Corrupted placeholder files — confined to `Legal_Compliance/CLO/*_Repository/`
139 files (pdf/docx/pptx/xlsx) under `Legal_Compliance/CLO/Nova_CLO_*_Repository/` folders
are not real documents — their file content is literally just the filename typed out as
plain text (e.g. `26_Trade_Compliance_Control_Testing_Workpapers.xlsx` contains the bytes
`26 Trade...`), instead of a valid PDF/OOXML binary. Confirmed via magic-byte check
(`%PDF-` / `PK\x03\x04` header). This is a source-data problem — present in the original
`data.zip`, not something introduced by our extraction/scoping scripts.

Affected folders (all under `Legal_Compliance/CLO/`):
- `Nova_CLO_Export_Control_Repository/` (+ duplicate `Nova_CLO_Export_Control_Repository (1)/`)
- `Nova_CLO_Compliance_Repository/`
- `Nova_CLO_Corporate_Law_Repository/`
- `Nova_CLO_Data_Privacy_Repository/`
- `Nova_CLO_Commercial_Contracts_Repository (1)/`

Not affected: the `NOVA_CLO_*_Subfolder (N)/` folders in the same directory (Intellectual
Property, Legal Reports, Litigation, Regulatory Affairs) — those files are real.

**Default handling (unless told otherwise):** the ingestion pipeline will detect these via
the same magic-byte check, skip them, and log each one to a `failed_files` report at the
end of the run — rather than silently dropping them or crashing the batch. Legal_Compliance
will end up under-represented until real source files are provided.

## 2. Duplicate folders
`Nova_CLO_Export_Control_Repository/` and `Nova_CLO_Export_Control_Repository (1)/` contain
byte-identical filenames/content (both corrupted placeholders in this case). The pipeline
will de-duplicate by content hash so identical files don't produce duplicate chunks/vectors,
but the `(1)` folder should ideally be removed at the source.

## 3. Everything else checked clean
Full-corpus magic-byte validation ran on all 4,273 in-scope pdf/docx/pptx/xlsx files
(csv/txt/md have no binary signature to check). Outside of `Legal_Compliance/CLO`, zero
corrupted files found.

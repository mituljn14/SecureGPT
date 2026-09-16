"""
document_registry.py

STEP 2 — Document Registry ONLY.

Given one or more document_id values (as returned by the Retriever
in Step 1), look them up in the existing document_registry.db
SQLite database and return the full registry row, most importantly
source_path.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict


# Resolve next to this module rather than the process working directory.
# As a bare relative path this raised at import time, because the module is
# imported long before coding_agent_node chdir()s into this folder.
DB_PATH = str(Path(__file__).resolve().parent / "document_registry.db")


class DocumentRegistry:
    """Legacy v1 lookup of document_id -> source_path.

    Optional as of v2: the securegpt_v2 payload carries `source_path` on
    every chunk, so node_retrieve resolves the file directly and this is
    consulted only as a fallback (and by the filename fast-path). A
    missing or unreadable database therefore degrades to "no match"
    instead of taking the whole agent down at import.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        if not Path(db_path).exists():
            print(f"[DocumentRegistry] {db_path} not found -- registry lookups "
                  f"disabled; using source_path from the v2 payload instead.")
            return
        try:
            uri = f"file:{Path(db_path).as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print(f"[DocumentRegistry] could not open {db_path}: {e} -- "
                  f"registry lookups disabled.")
            self.conn = None

    def get_by_document_id(self, document_id: str) -> Optional[dict]:
        """Looks up a single document_id."""
        if self.conn is None:
            return None
        cursor = self.conn.execute(
            "SELECT document_id, file_name, file_type, source_hash, "
            "source_path, chunk_count, created_at "
            "FROM documents WHERE document_id = ?",
            (document_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_file_name(self, file_name: str) -> Optional[dict]:
        """Looks up a document by exact or partial filename."""
        if self.conn is None:
            return None
        cursor = self.conn.execute(
            "SELECT document_id, file_name, file_type, source_hash, "
            "source_path, chunk_count, created_at "
            "FROM documents WHERE file_name = ? OR file_name LIKE ? LIMIT 1",
            (file_name, f"%{file_name}%"),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_many(self, document_ids: list) -> dict:
        """Looks up multiple document_ids in one query."""
        if self.conn is None:
            return {}
        if not document_ids:
            return {}

        unique_ids = list(set(document_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        cursor = self.conn.execute(
            f"SELECT document_id, file_name, file_type, source_hash, "
            f"source_path, chunk_count, created_at "
            f"FROM documents WHERE document_id IN ({placeholders})",
            unique_ids,
        )
        results = {row["document_id"]: dict(row) for row in cursor.fetchall()}
        return results

    def verify_source_hash(self, document_id: str, expected_hash: str) -> bool:
        """Confirms the source_hash matches."""
        record = self.get_by_document_id(document_id)
        if not record:
            return False
        return record.get("source_hash") == expected_hash

    def close(self):
        if self.conn is not None:
            self.conn.close()
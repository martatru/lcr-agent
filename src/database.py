"""
SQLite database management module for LCR Biocuration Agent.

Handles database initialization, connection pooling, and CRUD operations
for processed papers and LCR annotations.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "lcr_annotations.db"


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional scope around a database connection."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database tables for processed papers and LCR annotations if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_papers (
                paper_id TEXT PRIMARY KEY,
                doi TEXT,
                pmid TEXT,
                file_name TEXT,
                file_hash TEXT,
                model_name TEXT,
                prompt_version TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lcr_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                doi TEXT,
                protein_name TEXT,
                organism TEXT,
                start_pos TEXT,
                end_pos TEXT,
                binding_target TEXT,
                proposed_function TEXT,
                evidence TEXT,
                evidence_verified INTEGER,
                curation_status TEXT,
                FOREIGN KEY (paper_id) REFERENCES processed_papers (paper_id)
            )
            """
        )
        conn.commit()


def is_paper_processed(paper_id: str) -> bool:
    """Check if a paper has already been processed and stored in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processed_papers WHERE paper_id = ?", (paper_id,)
        )
        return cursor.fetchone() is not None


def save_paper_results(
    paper_id: str,
    doi: str,
    pmid: str,
    file_name: str,
    file_hash: str,
    annotations: List[Any],
    model_name: str,
    prompt_version: str,
) -> None:
    """Save processed paper metadata and its extracted LCR annotations into SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO processed_papers 
            (paper_id, doi, pmid, file_name, file_hash, model_name, prompt_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (paper_id, doi, pmid, file_name, file_hash, model_name, prompt_version),
        )

        # Clear existing annotations for this paper before re-inserting
        cursor.execute("DELETE FROM lcr_annotations WHERE paper_id = ?", (paper_id,))

        for ann in annotations:
            if hasattr(ann, "dict"):
                ann_dict = ann.dict()
            elif isinstance(ann, dict):
                ann_dict = ann
            else:
                ann_dict = vars(ann)

            cursor.execute(
                """
                INSERT INTO lcr_annotations 
                (paper_id, doi, protein_name, organism, start_pos, end_pos, binding_target, proposed_function, evidence, evidence_verified, curation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    doi,
                    ann_dict.get("protein_name", "Unknown"),
                    ann_dict.get("organism", "Unspecified"),
                    str(ann_dict.get("start_pos", "Unspecified")),
                    str(ann_dict.get("end_pos", "Unspecified")),
                    ann_dict.get("binding_target", "Unspecified"),
                    ann_dict.get("proposed_function", "Unspecified"),
                    ann_dict.get("evidence", "No evidence statement provided."),
                    1 if ann_dict.get("evidence_verified") else 0,
                    ann_dict.get("curation_status", "verified"),
                ),
            )
        conn.commit()


def get_paper_annotations(paper_id: str) -> List[Dict[str, Any]]:
    """Retrieve all LCR annotations associated with a specific paper ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM lcr_annotations WHERE paper_id = ?", (paper_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
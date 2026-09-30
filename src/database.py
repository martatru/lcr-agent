"""
SQLite Storage & Provenance tracking layer indexed by DOI/PMID.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_PATH = Path("data/lcr_annotations.db")


def init_db() -> None:
    """Initialize SQLite database schema and run migrations if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_papers (
                paper_id TEXT PRIMARY KEY,
                doi TEXT,
                pmid TEXT,
                file_name TEXT,
                file_hash TEXT,
                status TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Auto-migration: dodanie kolumny file_name, jeśli baza istniała wcześniej
        cursor.execute("PRAGMA table_info(processed_papers)")
        columns = [row[1] for row in cursor.fetchall()]
        if "file_name" not in columns:
            cursor.execute("ALTER TABLE processed_papers ADD COLUMN file_name TEXT")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lcr_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT,
                doi TEXT,
                protein_name TEXT,
                organism TEXT,
                start_pos INTEGER,
                end_pos INTEGER,
                binding_target TEXT,
                proposed_function TEXT,
                evidence TEXT,
                evidence_verified INTEGER,
                curation_status TEXT,
                model_used TEXT,
                prompt_version TEXT,
                FOREIGN KEY (paper_id) REFERENCES processed_papers (paper_id)
            )
        """)
        conn.commit()


def is_paper_processed(paper_id: str) -> bool:
    """Check if a paper has already been processed successfully in database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processed_papers WHERE paper_id = ? AND status = 'COMPLETED'",
            (paper_id,)
        )
        return cursor.fetchone() is not None


def save_paper_results(
    paper_id: str,
    doi: Optional[str],
    pmid: Optional[str],
    file_name: str,
    file_hash: str,
    annotations: List[Dict[str, Any]],
    model_name: str,
    prompt_version: str
) -> None:
    """Save paper metadata and its extracted LCR annotations atomically."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO processed_papers (paper_id, doi, pmid, file_name, file_hash, status)
            VALUES (?, ?, ?, ?, ?, 'COMPLETED')
        """, (paper_id, doi or "", pmid or "", file_name, file_hash))

        for ann in annotations:
            cursor.execute("""
                INSERT INTO lcr_annotations (
                    paper_id, doi, protein_name, organism, start_pos, end_pos,
                    binding_target, proposed_function, evidence, evidence_verified,
                    curation_status, model_used, prompt_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper_id,
                doi or "",
                ann.get("protein_name"),
                ann.get("organism"),
                ann.get("start_of_annotation"),
                ann.get("end_of_annotation"),
                ann.get("binding_target"),
                ann.get("proposed_function"),
                ann.get("evidence"),
                1 if ann.get("evidence_verified") else 0,
                ann.get("curation_status"),
                model_name,
                prompt_version
            ))
        conn.commit()
"""
SQLite database management module for storing LCR annotations, processed papers,
and persistent caches for UniProt and PlaToLoCo API responses.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "lcr_annotations.db"


def get_db_connection() -> sqlite3.Connection:
    """Establish and return a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables including papers, annotations, and caches."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Processed papers table
        cursor.execute("""
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
        """)

        # LCR annotations table
        cursor.execute("""
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
                evidence_verified INTEGER DEFAULT 0,
                curation_status TEXT DEFAULT 'manual_check',
                FOREIGN KEY (paper_id) REFERENCES processed_papers(paper_id)
            )
        """)

        # Persistent cache for UniProt metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uniprot_cache (
                cache_key TEXT PRIMARY KEY,
                uniprot_id TEXT,
                gene_name TEXT,
                full_name TEXT,
                length INTEGER,
                sequence TEXT,
                go_terms_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Persistent cache for PlaToLoCo prediction results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS platoloco_cache (
                sequence_hash TEXT PRIMARY KEY,
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


def get_cached_uniprot(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve UniProt metadata from persistent SQLite cache."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT uniprot_id, gene_name, full_name, length, sequence, go_terms_json FROM uniprot_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "uniprot_id": row["uniprot_id"],
                "gene_name": row["gene_name"],
                "full_name": row["full_name"],
                "length": row["length"],
                "sequence": row["sequence"],
                "go_terms": json.loads(row["go_terms_json"]) if row["go_terms_json"] else [],
            }
    return None


def save_cached_uniprot(cache_key: str, data: Dict[str, Any]) -> None:
    """Save UniProt metadata to persistent SQLite cache."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO uniprot_cache 
            (cache_key, uniprot_id, gene_name, full_name, length, sequence, go_terms_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                data.get("uniprot_id", "N/A"),
                data.get("gene_name", ""),
                data.get("full_name", ""),
                data.get("length", 0),
                data.get("sequence", ""),
                json.dumps(data.get("go_terms", []), ensure_ascii=False),
            ),
        )
        conn.commit()


def get_cached_platoloco(sequence_hash: str) -> Optional[Dict[str, List[Dict[str, int]]]]:
    """Retrieve PlaToLoCo predictions from persistent SQLite cache."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT results_json FROM platoloco_cache WHERE sequence_hash = ?",
            (sequence_hash,),
        )
        row = cursor.fetchone()
        if row and row["results_json"]:
            return json.loads(row["results_json"])
    return None


def save_cached_platoloco(sequence_hash: str, results: Dict[str, List[Dict[str, int]]]) -> None:
    """Save PlaToLoCo predictions to persistent SQLite cache."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO platoloco_cache (sequence_hash, results_json)
            VALUES (?, ?)
            """,
            (sequence_hash, json.dumps(results, ensure_ascii=False)),
        )
        conn.commit()


def is_paper_processed(paper_id: str) -> bool:
    """Check whether a paper has already been processed in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed_papers WHERE paper_id = ?", (paper_id,))
        return cursor.fetchone() is not None


def save_paper_results(
    paper_id: str,
    doi: str,
    pmid: str,
    file_name: str,
    file_hash: str,
    annotations: List[Dict[str, Any]],
    model_name: str = "Groq-Cascade",
    prompt_version: str = "v1.0",
) -> None:
    """Save paper processing metadata and extracted LCR annotations to database."""
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

        for ann in annotations:
            cursor.execute(
                """
                INSERT INTO lcr_annotations 
                (paper_id, doi, protein_name, organism, start_pos, end_pos, binding_target, proposed_function, evidence, evidence_verified, curation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    doi,
                    ann.get("protein_name", ""),
                    ann.get("organism", ""),
                    str(ann.get("start_pos", "")) if ann.get("start_pos") is not None else None,
                    str(ann.get("end_pos", "")) if ann.get("end_pos") is not None else None,
                    ann.get("binding_target", ""),
                    ann.get("proposed_function", ""),
                    ann.get("evidence", ""),
                    1 if ann.get("evidence_verified") else 0,
                    ann.get("curation_status", "manual_check"),
                ),
            )

        conn.commit()
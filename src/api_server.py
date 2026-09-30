"""
FastAPI application for LCR Annotation queries and Biocuration API.
"""

import sqlite3
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from database import DB_PATH, init_db

app = FastAPI(
    title="LCR-Agent Biocuration API",
    version="1.0.0",
    description="REST API for functional Low-Complexity Region (LCR) annotations in proteins."
)


@app.on_event("startup")
def startup() -> None:
    """Initialize database on server startup."""
    init_db()


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "online"}


@app.get("/annotations/{paper_id}", response_model=List[Dict[str, Any]])
def get_annotations_for_paper(paper_id: str) -> List[Dict[str, Any]]:
    """Retrieve extracted LCR annotations for a given DOI or PMID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM lcr_annotations WHERE paper_id = ?", (paper_id,))
        rows = cursor.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Paper not found or no annotations available")
        return [dict(row) for row in rows]
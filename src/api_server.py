"""
FastAPI REST Server for LCR Biocuration Agent.

Provides endpoints to manage PDF papers, trigger extraction pipelines,
retrieve database annotations, serve generated HTML reports, and host
the Web Dashboard interface.
"""

import asyncio
import datetime
import hashlib
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Adjust Python search path to resolve relative module imports
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import get_db_connection, init_db, save_paper_results
from generate_report import generate_html_report
from llm_client import LightLLMClient
from main import compute_file_hash, process_pdf_file

app = FastAPI(
    title="LCR Agent API",
    description="REST API for Low-Complexity Region biocuration and report generation.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = SRC_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup_event() -> None:
    """Initialize SQLite database structure on server startup."""
    init_db()


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> FileResponse:
    """Serve the interactive Web Dashboard homepage."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint to verify backend service status."""
    return {"status": "ok", "message": "LCR Agent API is running"}


@app.get("/models")
def list_models() -> List[str]:
    """Return available LLM models configured in LightLLMClient."""
    client = LightLLMClient()
    return client.models


@app.get("/papers")
def list_papers() -> List[Dict[str, Any]]:
    """Fetch all processed papers stored in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT paper_id, doi, pmid, file_name, model_name, processed_at 
            FROM processed_papers 
            ORDER BY processed_at DESC
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


@app.delete("/papers/clear")
def clear_papers() -> Dict[str, str]:
    """Clear all processed papers and annotations from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM lcr_annotations")
        cursor.execute("DELETE FROM processed_papers")
        conn.commit()
    return {"message": "All database records cleared successfully"}


@app.get("/annotations")
def get_annotations(paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve LCR annotations filtered optionally by paper_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT 
                a.id,
                a.paper_id,
                a.doi,
                COALESCE(p.file_name, 'N/A') AS file_name,
                a.protein_name,
                a.organism,
                a.start_pos,
                a.end_pos,
                a.binding_target,
                a.proposed_function,
                a.evidence,
                a.evidence_verified,
                a.curation_status
            FROM lcr_annotations a
            LEFT JOIN processed_papers p ON a.paper_id = p.paper_id
        """
        params: List[Any] = []
        if paper_id:
            query += " WHERE a.paper_id = ?"
            params.append(paper_id)

        query += " ORDER BY a.id ASC"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        records = []
        for row in rows:
            rec = dict(row)
            rec["evidence_verified"] = bool(rec["evidence_verified"])
            records.append(rec)
        return records


class ReportRequest(BaseModel):
    paper_id: Optional[str] = None


@app.post("/reports/generate")
def create_report(request: ReportRequest) -> Dict[str, Any]:
    """Trigger generation of HTML biocuration report with clean lowercase timestamped filename."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if request.paper_id:
        clean_id = re.sub(r"[^a-zA-Z0-9]", "_", request.paper_id).lower()
        output_filename = f"lcr_report_{clean_id}_{timestamp}.html"
    else:
        output_filename = f"lcr_report_summary_{timestamp}.html"

    reports_dir = PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / output_filename

    generated_path = generate_html_report(
        paper_id=request.paper_id,
        output_html=str(output_path),
    )

    if not generated_path or not Path(generated_path).exists():
        raise HTTPException(
            status_code=400,
            detail="Failed to generate report or no data found for this paper",
        )

    return {
        "message": "Report generated successfully",
        "file_name": output_filename,
        "download_url": f"/reports/download/{output_filename}",
    }


@app.get("/reports/download/{file_name}")
def download_report(file_name: str) -> FileResponse:
    """Serve generated HTML report file for inline browser viewing."""
    file_path = PROJECT_ROOT / "data" / "reports" / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(path=file_path, media_type="text/html")


def extract_doi_from_pdf(pdf_path: Path) -> Optional[str]:
    """Attempt to extract a DOI string from the first 2 pages of a PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages[:2]:
            t = page.extract_text()
            if t:
                text += t + "\n"

        doi_pattern = r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"
        match = re.search(doi_pattern, text)
        if match:
            return match.group(1).rstrip(".,;")
    except Exception:
        pass
    return None


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload raw PDF paper instantly and register metadata without blocking on AI pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    raw_dir = PROJECT_ROOT / "data" / "raw_pdfs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / file.filename

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_doi = extract_doi_from_pdf(destination)
        if not extracted_doi:
            clean_name = Path(file.filename).stem.lower().replace(" ", "_")
            extracted_doi = f"10.5555/local_{clean_name}"

        paper_id = extracted_doi
        file_bytes = destination.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO processed_papers 
                (paper_id, doi, pmid, file_name, file_hash, model_name, prompt_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    extracted_doi,
                    "N/A",
                    file.filename,
                    file_hash,
                    "Pending",
                    "v1.0",
                ),
            )
            conn.commit()

        return {
            "message": "PDF uploaded successfully. Click Run AI Pipeline in the table below.",
            "paper_id": paper_id,
            "file_name": file.filename,
        }
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Upload failed: {str(err)}"
        )


class PipelineRequest(BaseModel):
    model_name: Optional[str] = None


@app.post("/papers/{paper_id:path}/run-pipeline")
async def run_pipeline(paper_id: str, request: PipelineRequest) -> Dict[str, Any]:
    """Execute the AI extraction pipeline for a specific registered paper."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_name, file_hash FROM processed_papers WHERE paper_id = ?", (paper_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found in database")
        file_name = row["file_name"]
        file_hash = row["file_hash"]

    pdf_path = PROJECT_ROOT / "data" / "raw_pdfs" / file_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    try:
        client = LightLLMClient(max_concurrent=1)
        if request.model_name and request.model_name in client.models:
            client.models = [request.model_name] + [m for m in client.models if m != request.model_name]

        extracted_doi, annotations = await process_pdf_file(pdf_path, client)

        save_paper_results(
            paper_id=paper_id,
            doi=extracted_doi or paper_id,
            pmid="",
            file_name=file_name,
            file_hash=file_hash,
            annotations=annotations,
            model_name=request.model_name or client.models[0],
            prompt_version="v1.0",
        )

        return {
            "message": "AI pipeline executed successfully",
            "paper_id": paper_id,
        }
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {str(err)}"
        )
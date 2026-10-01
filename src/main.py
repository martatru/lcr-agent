"""
Core pipeline orchestration module for PDF text extraction, DOI identification, and LLM biocuration.
"""

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure 'src' directory is in Python path for smooth execution from any directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import get_paper_annotations, init_db, is_paper_processed, save_paper_results
from generate_report import generate_html_report
from llm_client import LightLLMClient
from text_extractor import (
    chunk_sections_by_paragraphs,
    deduplicate_annotations,
    extract_doi_from_text,
    extract_tagged_sections,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1.0"
PROMPT_LCR = """Your task is to go through the text provided, read it thoroughly, and identify the exact presence or close-remote relationship of the following keywords: low-complexity, low-complexity region(s), 
LCR, repeat(s), tandem repeat(s), repetitive, intrinsically disordered region(s), IDP, IDR, and any other related terms that indicate the presence of a low-complexity region (LCR) or low-complexity domain (LCD) 
in a protein. 

Guidelines:
1. Extract all instance candidates of the specified keywords.
2. Extract all mentioned proteins or genes, including their names and identifiers (e.g., UniProt ID, gene symbol).
3. Extract any presence of a function, regulation, or interaction mentioned in the text.
4. For each interaction, function, or regulation role, map it to the corresponding protein or gene if possible. ELSE, flag it as 'Unspecified'.
5. DO NOT ASSUME that the presence of a keyword, a function, or a protein implies an actual relationship between them.
6. If present, map coordinates to the LCRs or proteins mentioned. If coordinates are not present, flag them as 'Unspecified'. DO NOT ASSUME coordinates 
based on the text.
7. 'evidence' MUST be an exact verbatim sentence from the text proving the LCR and its relationship to the identified protein or gene, and function IF 
the relationship EXISTS.
"""


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for resume and provenance tracking."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


async def process_pdf_file(
    pdf_path: Path, client: LightLLMClient, progress_callback=None
) -> tuple[str, list[dict]]:
    """Extract text by tagged sections, identify DOI, chunk, and query LLM API with progress reporting."""
    logger.info("Processing file: %s", pdf_path.name)

    sections = extract_tagged_sections(str(pdf_path))
    if not sections:
        logger.error("Failed to extract text from PDF: %s", pdf_path.name)
        return "", []

    full_text = "\n".join([s["text"] for s in sections])
    extracted_doi = extract_doi_from_text(full_text) or ""
    if extracted_doi:
        logger.info("Identified DOI for %s: %s", pdf_path.name, extracted_doi)
    else:
        logger.warning("No DOI found in text for %s. Falling back to filename.", pdf_path.name)

    chunks = chunk_sections_by_paragraphs(sections, max_chunk_size=8000)
    total_chunks = len(chunks)
    raw_annotations = []
    debug_logs = []

    for idx, chunk in enumerate(chunks, start=1):
        if progress_callback:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(idx, total_chunks, f"Processing chunk {idx}/{total_chunks}")
            else:
                progress_callback(idx, total_chunks, f"Processing chunk {idx}/{total_chunks}")

        result = await client.generate_lcr_annotations(PROMPT_LCR, chunk)

        status = result.get("status")
        annotations = result.get("annotations", [])

        if status in ("ok", "empty"):
            raw_annotations.extend(annotations)
        elif status == "failed":
            logger.error("Chunk %d failed on %s: %s", idx, pdf_path.name, result.get("error"))

        debug_logs.append({
            "chunk_index": idx,
            "chunk_length": len(chunk),
            "status": status,
            "raw_extracted_count": len(annotations),
            "raw_annotations": annotations,
        })

    debug_dir = PROJECT_ROOT / "data" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    json_debug_file = debug_dir / f"{pdf_path.stem}_debug.json"
    with open(json_debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_logs, f, indent=2, ensure_ascii=False)

    clean_annotations = deduplicate_annotations(raw_annotations)
    return extracted_doi, clean_annotations
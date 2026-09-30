"""
Core pipeline orchestration module for PDF text extraction, DOI identification, and LLM biocuration.
"""

import asyncio
import json
import logging
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from generate_report import generate_html_report
from llm_client import LightLLMClient
from text_extractor import (
    extract_tagged_sections,
    chunk_sections_by_paragraphs,
    deduplicate_annotations,
    extract_doi_from_text,
)
from database import init_db, save_paper_results, is_paper_processed

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1.0"
PROMPT_LCR = """Your task is to go through the text provided, read it thoroughly, and identify the exact presence or close-remote relationship of the following keywords: low-complexity, low-complexity region(s), 
LCR, repeat(s), tandem repeat(s), repetitive, instrinsically disordered region(s), IDP, IDR, and any other related terms that indicate the presence of a low-complexity region (LCR) or low-complexity domain (LCD) 
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
    pdf_path: Path, client: LightLLMClient
) -> tuple[str, list[dict]]:
    """Extract text by tagged sections, identify DOI, chunk, and query LLM API."""
    logger.info("Processing file: %s", pdf_path.name)

    sections = extract_tagged_sections(str(pdf_path))
    if not sections:
        logger.error("Failed to extract text from PDF: %s", pdf_path.name)
        return "", []

    # Extract DOI from the full extracted text
    full_text = "\n".join([s["text"] for s in sections])
    extracted_doi = extract_doi_from_text(full_text) or ""
    if extracted_doi:
        logger.info("Identified DOI for %s: %s", pdf_path.name, extracted_doi)
    else:
        logger.warning("No DOI found in text for %s. Falling back to filename.", pdf_path.name)

    chunks = chunk_sections_by_paragraphs(sections, max_chunk_size=8000)
    raw_annotations = []
    debug_logs = []

    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Processing chunk %d/%d for %s...", idx, len(chunks), pdf_path.name)
        result = await client.generate_lcr_annotations(PROMPT_LCR, chunk)

        status = result.get("status")
        annotations = result.get("annotations", [])

        if status in ("ok", "empty"):
            raw_annotations.extend(annotations)
        elif status == "failed":
            logger.error(
                "Chunk %d failed on %s: %s", idx, pdf_path.name, result.get("error")
            )

        debug_logs.append({
            "chunk_index": idx,
            "chunk_length": len(chunk),
            "status": status,
            "raw_extracted_count": len(annotations),
            "raw_annotations": annotations
        })

    # Save debug logs
    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    json_debug_file = debug_dir / f"{pdf_path.stem}_debug.json"
    with open(json_debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_logs, f, indent=2, ensure_ascii=False)

    clean_annotations = deduplicate_annotations(raw_annotations)
    return extracted_doi, clean_annotations


async def main():
    """Main execution workflow for processing PDF files directly."""
    init_db()

    input_dir = Path("/home/marta/Pulpit/lcr-agent/data/paper_pdf")
    output_dir = Path("/home/marta/Pulpit/lcr-agent/data/processed")
    reports_dir = Path("/home/marta/Pulpit/lcr-agent/data/reports")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_title_input = input(
        "Enter report title/filename (e.g., benchmark_lcr_report): "
    ).strip()

    if not report_title_input:
        report_title_input = "lcr_biocuration_report"

    if not report_title_input.lower().endswith(".html"):
        report_filename = f"{report_title_input}.html"
    else:
        report_filename = report_title_input

    html_report_file = reports_dir / report_filename
    jsonl_output_file = (
        output_dir / f"{Path(report_filename).stem}_results.jsonl"
    )

    if not input_dir.exists():
        logger.error("PDF directory does not exist: %s", input_dir)
        return

    pdf_files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
        ]
    )

    if not pdf_files:
        logger.warning("No valid PDF files found in directory: %s", input_dir)
        return

    logger.info("Found %d PDF files in %s", len(pdf_files), input_dir)

    client = LightLLMClient(max_concurrent=1)
    await client.validate_models()

    results = []

    for pdf_file in pdf_files:
        file_hash = compute_file_hash(pdf_file)

        # Extract DOI and annotations
        extracted_doi, annotations = await process_pdf_file(pdf_file, client)

        # Determine unique paper identifier (DOI preferred over filename)
        paper_id = extracted_doi if extracted_doi else pdf_file.stem

        if is_paper_processed(paper_id):
            logger.info("Skipping already processed paper: %s", paper_id)
            continue

        # Save results incrementally into SQLite database
        save_paper_results(
            paper_id=paper_id,
            doi=extracted_doi,
            pmid="",
            file_name=pdf_file.name,
            file_hash=file_hash,
            annotations=annotations,
            model_name="Groq-Cascade",
            prompt_version=PROMPT_VERSION,
        )

        if annotations:
            entry = {
                "file": pdf_file.name,
                "doi": extracted_doi,
                "source_id": paper_id,
                "annotations": annotations
            }
            results.append(entry)

            with open(jsonl_output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Saved raw results to: %s", jsonl_output_file)

    logger.info("Generating HTML biocuration report...")
    if jsonl_output_file.exists():
        generate_html_report(
            input_file=str(jsonl_output_file), output_html=str(html_report_file)
        )
        logger.info("Pipeline finished! Report saved at: %s", html_report_file)
    else:
        logger.warning("No new annotations extracted. HTML report omitted.")


if __name__ == "__main__":
    asyncio.run(main())
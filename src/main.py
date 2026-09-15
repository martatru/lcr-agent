"""
Core pipeline orchestration module for PDF text extraction and LLM biocuration.
"""

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from generate_report import generate_html_report
from llm_client import LightLLMClient
from text_extractor import parse_pdf, extract_core_results_only, chunk_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


# =============================================================================
# OLD TEXT FILE PROCESSING FUNCTION (PRESERVED & COMMENTED OUT)
# =============================================================================
# async def process_text_file(
#     text_path: Path, client: LightLLMClient
# ) -> list[dict]:
#     """Process a pre-extracted text file, chunk text, and query LLM API."""
#     logger.info("Processing file: %s", text_path)
#
#     try:
#         clean_text = text_path.read_text(encoding="utf-8")
#     except Exception as err:
#         logger.error("Error reading text file %s: %s", text_path, err)
#         return []
#
#     if not clean_text.strip():
#         logger.warning("File is empty: %s", text_path)
#         return []
#
#     chunks = chunk_text(clean_text, chunk_size=12000, overlap=2000)
#     all_annotations = []
#     debug_logs = []
#
#     for idx, chunk in enumerate(chunks):
#         logger.info(
#             "Processing chunk %d/%d for %s...",
#             idx + 1,
#             len(chunks),
#             text_path.name,
#         )
#         annotations = await client.generate_lcr_annotations(PROMPT_LCR, chunk)
#
#         all_annotations.extend(annotations)
#
#         debug_logs.append(
#             {
#                 "chunk_index": idx,
#                 "chunk_length": len(chunk),
#                 "raw_extracted_count": len(annotations),
#                 "raw_annotations": annotations,
#             }
#         )
#
#         if idx < len(chunks) - 1:
#             await asyncio.sleep(20)
#
#     debug_dir = Path("data/debug")
#     debug_dir.mkdir(parents=True, exist_ok=True)
#     file_stem = text_path.stem
#
#     json_debug_file = debug_dir / f"{file_stem}_debug.json"
#     with open(json_debug_file, "w", encoding="utf-8") as f:
#         json.dump(debug_logs, f, indent=2, ensure_ascii=False)
#
#     return all_annotations
# =============================================================================


async def process_pdf_file(
    pdf_path: Path, client: LightLLMClient
) -> list[dict]:
    """Extract text directly from PDF, clean core sections, chunk, and query LLM API."""
    logger.info("Processing file: %s", pdf_path)

    raw_text = parse_pdf(str(pdf_path))
    if not raw_text.strip():
        logger.error("Failed to extract text from PDF: %s", pdf_path)
        return []

    clean_text = extract_core_results_only(raw_text)
    if not clean_text.strip():
        logger.warning("Extracted core text is empty for file: %s", pdf_path)
        return []

    chunks = chunk_text(clean_text, chunk_size=12000, overlap=2000)
    all_annotations = []
    debug_logs = []

    for idx, chunk in enumerate(chunks):
        logger.info(
            "Processing chunk %d/%d for %s...",
            idx + 1,
            len(chunks),
            pdf_path.name,
        )
        annotations = await client.generate_lcr_annotations(PROMPT_LCR, chunk)

        all_annotations.extend(annotations)

        debug_logs.append(
            {
                "chunk_index": idx,
                "chunk_length": len(chunk),
                "raw_extracted_count": len(annotations),
                "raw_annotations": annotations,
            }
        )

        if idx < len(chunks) - 1:
            await asyncio.sleep(20)

    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    file_stem = pdf_path.stem

    json_debug_file = debug_dir / f"{file_stem}_debug.json"
    with open(json_debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_logs, f, indent=2, ensure_ascii=False)

    return all_annotations


async def main():
    """Main execution workflow for processing PDF files directly."""
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
    results = []

    for pdf_file in pdf_files:
        annotations = await process_pdf_file(pdf_file, client)
        if annotations:
            results.append({"file": pdf_file.name, "annotations": annotations})

    with open(jsonl_output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Saved raw results to: %s", jsonl_output_file)

    logger.info("Generating HTML biocuration report...")
    generate_html_report(
        input_file=str(jsonl_output_file), output_html=str(html_report_file)
    )

    logger.info("Pipeline finished! Report saved at: %s", html_report_file)


if __name__ == "__main__":
    asyncio.run(main())
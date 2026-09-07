"""
Core pipeline orchestration module for PDF text extraction and LLM biocuration.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from llm_client import LightLLMClient
from pdf_parser import parse_pdf
from text_processor import prepare_full_text, chunk_text
from validator import validate_lcr_annotations
from generate_report import generate_html_report
from post_processor import process_results

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT_LCR = """You are an expert biocuration AI. Your task is to extract Low Complexity Regions (LCRs/IDRs/PLDs), their coordinate ranges, and associated binding interactions from scientific text.

Guidelines:
1. **Coordinate Extraction (CRITICAL)**: Scrutinize the text for any residue numbers, ranges, or amino acid positions (e.g., "residues 904–1297", "amino acids 50-100", "positions 12-61"). If numerical coordinates are present, you MUST extract the start integer into 'start_of_annotation' and the end integer into 'end_of_annotation'. Do not leave them as 'Unspecified' if numbers are stated.
2. **Binding Interaction Naming**: In 'binding_target', specify the precise interaction type using the suffix '-binding' (e.g., 'protein-binding', 'RNA-binding', 'DNA-binding'). Never use isolated nouns like 'Protein'. If no target is mentioned, set to 'Unspecified'.
3. **Verified Category**: Set 'curation_status' to 'Verified' ONLY if explicit numerical coordinates AND experimental binding evidence are found.
4. **Requires Manual Check Category**: Set 'curation_status' to 'Requires Manual Check' if numerical coordinates are completely missing from the text or the mention is purely qualitative.
5. **Evidence Rule**: 'evidence' must contain an exact verbatim sentence from the text proving the statement.
"""

async def process_pdf_file(pdf_path: str, client: LightLLMClient) -> list[dict]:
    """Processes a single PDF file, chunks text, queries Groq API, and validates results."""
    logger.info("Processing file: %s", pdf_path)

    raw_text = parse_pdf(pdf_path)
    if not raw_text.strip():
        logger.error("Failed to extract text from PDF: %s", pdf_path)
        return []

    clean_text = prepare_full_text(raw_text)

    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(pdf_path).stem

    text_debug_file = debug_dir / f"{pdf_stem}_text.txt"
    with open(text_debug_file, "w", encoding="utf-8") as f:
        f.write(clean_text)
    logger.info("Saved extracted text (%d chars) to: %s", len(clean_text), text_debug_file)

    chunks = chunk_text(clean_text, chunk_size=12000, overlap=2000)
    all_annotations = []
    debug_logs = []

    for idx, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d...", idx + 1, len(chunks))
        annotations = await client.generate_lcr_annotations(PROMPT_LCR, chunk)

        valid_annotations = validate_lcr_annotations(annotations, chunk)

        debug_logs.append({
            "chunk_index": idx,
            "chunk_length": len(chunk),
            "raw_extracted_count": len(annotations),
            "valid_count": len(valid_annotations),
            "raw_annotations": annotations,
            "valid_annotations": valid_annotations
        })

        if valid_annotations:
            all_annotations.extend(valid_annotations)

        if idx < len(chunks) - 1:
            await asyncio.sleep(20)

    json_debug_file = debug_dir / f"{pdf_stem}_debug.json"
    with open(json_debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_logs, f, indent=2, ensure_ascii=False)

    return all_annotations


async def main():
    """Main execution workflow for processing all PDF files in input directory."""
    input_dir = Path("data/raw_pdfs")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_output_file = output_dir / "final_results.jsonl"
    html_report_file = output_dir / "lcr_biocuration_report.html"

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No .pdf files found in directory %s", input_dir)
        return

    client = LightLLMClient(max_concurrent=1)
    results = []

    for pdf_file in pdf_files:
        annotations = await process_pdf_file(str(pdf_file), client)
        if annotations:
            results.append({
                "file": pdf_file.name,
                "annotations": annotations
            })

    with open(jsonl_output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Saved raw results to: %s", jsonl_output_file)

    process_results(
        input_file=str(jsonl_output_file),
        verified_output_file=str(output_dir / "verified_lcrs.json"),
        qualitative_output_file=str(output_dir / "qualitative_mentions.json"),
        qualitative_csv_file=str(output_dir / "qualitative_mentions.csv")
    )

    logger.info("Generating HTML biocuration report...")
    generate_html_report(
        input_file=str(jsonl_output_file),
        output_html=str(html_report_file)
    )

    logger.info("Pipeline finished! View report at: %s", html_report_file)


if __name__ == "__main__":
    asyncio.run(main())
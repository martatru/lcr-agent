"""
Benchmark Text Extraction Script for LCR-Agent Pipeline.

Parses downloaded PDF files from raw input directory, applies header-based section
filtering (isolating Results, Discussion, Conclusions), and persists plain text files
to data/benchmark/extracted_text/ for reusable benchmark execution.
"""

import logging
from pathlib import Path

from pdf_parser import parse_pdf
from text_processor import extract_core_results_only, prepare_full_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_benchmark_texts(
    input_dir: str = "data/benchmark/papers",
    output_dir: str = "data/benchmark/extracted_text",
    filter_sections: bool = True
) -> None:
    """
    Parses all PDFs in input_dir and saves extracted clean text to output_dir.

    Args:
        input_dir: Directory containing input PDF files.
        output_dir: Target directory for extracted plain text (.txt) files.
        filter_sections: If True, uses extract_core_results_only to strip Methods/Refs.
                         If False, saves full cleaned PDF text.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in directory '%s'!", input_dir)
        return

    logger.info("Found %d PDF files in '%s'. Starting batch extraction...", len(pdf_files), input_dir)

    success_count = 0
    fail_count = 0

    for pdf_file in pdf_files:
        logger.info("Processing: %s...", pdf_file.name)
        raw_text = parse_pdf(str(pdf_file))

        if not raw_text.strip():
            logger.error("Failed to extract raw text from PDF: %s", pdf_file.name)
            fail_count += 1
            continue

        if filter_sections:
            processed_text = extract_core_results_only(raw_text)
        else:
            processed_text = prepare_full_text(raw_text)

        # Fallback to full text if section filtering yielded insufficient text
        if not processed_text.strip():
            logger.warning("Section extraction returned empty string for %s; using full text fallback.", pdf_file.name)
            processed_text = prepare_full_text(raw_text)

        out_txt_path = output_path / f"{pdf_file.stem}.txt"
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        logger.info("Saved extracted text (%d chars) -> %s", len(processed_text), out_txt_path)
        success_count += 1

    logger.info("==================================================")
    logger.info("BATCH TEXT EXTRACTION COMPLETED")
    logger.info("Successfully processed: %d/%d files", success_count, len(pdf_files))
    if fail_count > 0:
        logger.warning("Failed extractions: %d files", fail_count)
    logger.info("Extracted texts location: %s", output_path)
    logger.info("==================================================")


if __name__ == "__main__":
    extract_benchmark_texts()
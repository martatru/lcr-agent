"""
Benchmark text extraction runner.
Processes all PDFs in benchmark directory and saves filtered text outputs.
"""

import logging
from pathlib import Path
from text_extractor import parse_pdf, extract_core_results_only

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = Path("/home/marta/Pulpit/lcr-agent/data/benchmark/papers")
OUTPUT_DIR = Path("/home/marta/Pulpit/lcr-agent/data/benchmark/extracted_text")


def run_benchmark():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(INPUT_DIR.glob("*.pdf")))
    if not pdf_files:
        logger.error("Nie znaleziono plików .pdf w katalogu: %s", INPUT_DIR)
        return

    logger.info("Znaleziono %d plików PDF do przetworzenia.", len(pdf_files))

    total_raw_chars = 0
    total_clean_chars = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        raw_text = parse_pdf(str(pdf_path))
        if not raw_text.strip():
            logger.warning("[%d/%d] Błąd ekstrakcji lub pusty PDF: %s", idx, len(pdf_files), pdf_path.name)
            continue

        filtered_text = extract_core_results_only(raw_text)

        raw_len = len(raw_text)
        clean_len = len(filtered_text)
        total_raw_chars += raw_len
        total_clean_chars += clean_len

        out_file = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        out_file.write_text(filtered_text, encoding="utf-8")

        reduction = (1 - (clean_len / raw_len)) * 100 if raw_len > 0 else 0
        logger.info(
            "[%d/%d] %s: %d -> %d znaków (redukcja: %.1f%%)",
            idx, len(pdf_files), pdf_path.name, raw_len, clean_len, reduction
        )

    avg_reduction = (1 - (total_clean_chars / total_raw_chars)) * 100 if total_raw_chars > 0 else 0
    logger.info("---")
    logger.info("Zakończono! Zapisano pliki w: %s", OUTPUT_DIR)
    logger.info("Łącznie znaków: %d -> %d (Średnia redukcja szumu: %.1f%%)", total_raw_chars, total_clean_chars, avg_reduction)


if __name__ == "__main__":
    run_benchmark()
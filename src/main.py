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

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT_LCR = """You are a specialized biocuration AI. Your task is to extract all experimentally verified Low Complexity Regions (LCRs) present in the provided text.

Rules:
1. Extract numerical residue coordinates (start_of_annotation, end_of_annotation).
2. 'evidence' MUST be an exact verbatim sentence from the text as proof.
3. If positions are given as 'residues 145-180', start_of_annotation=145, end_of_annotation=180.
4. If no experimentally verified LCRs are found, return an empty array.
"""

async def process_pdf_file(pdf_path: str, client: LightLLMClient) -> list[dict]:
    logger.info("Processing file: %s", pdf_path)
    
    # 1. Wyciągnięcie tekstu z PDF (PyMuPDF z pdf_parser.py)
    raw_text = parse_pdf(pdf_path)
    if not raw_text.strip():
        logger.error("Failed to extract text from PDF: %s", pdf_path)
        return []

    # 2. Czyszczenie tekstu (usunięcie bibliografii, sklejanie przeniesień)
    clean_text = prepare_full_text(raw_text)

    # 3. Dzielenie na mniejsze chunki (22k znaków ~= 5.5k tokenów, żeby zmieścić się w limicie 8k TPM)
    chunks = chunk_text(clean_text, chunk_size=22000, overlap=3000)
    all_annotations = []
    debug_logs = []

    for idx, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d...", idx + 1, len(chunks))
        annotations = await client.generate_lcr_annotations(PROMPT_LCR, chunk)
        
        # Walidacja (czy wyciągnięty cytat istnieje fizycznie w tym chunku)
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

        # Odczekaj 6 sekund przed kolejnym zapytaniem, by nie przekroczyć darmowego limitu tokenów/minutę
        if idx < len(chunks) - 1:
            await asyncio.sleep(6)

    # Zapis szczegółowych logów do data/debug/
    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"{Path(pdf_path).stem}_debug.json"
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_logs, f, indent=2, ensure_ascii=False)

    return all_annotations

async def main():
    input_dir = Path("data/raw_pdfs")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "final_results.jsonl"

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

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Success! Saved %d results in %s", len(results), output_file)

if __name__ == "__main__":
    asyncio.run(main())
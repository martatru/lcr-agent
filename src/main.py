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

PROMPT_LCR = """You are an expert biocuration AI. Your task is to exhaustively extract ALL Low Complexity Regions (LCRs/LCDs/IDRs/PLDs) mentioned in the text—both those with explicit residue numbers and those described qualitatively.

Guidelines:
1. Extract every protein mentioned to contain or form an LCR/LCD (e.g., FUS, TIA1, hnRNPA1, hnRNPA2, CIRBP, RBM3, Sup35, TDP43, FMRP).
2. If numerical residue coordinates are given (e.g., 'residues 2-214'), extract them as strings in start_of_annotation and end_of_annotation. If no numbers are provided in the text, set them to 'Unspecified'.
3. 'evidence' MUST be an exact verbatim sentence from the text proving the LCR and its position/function.
4. In 'curator_note', state whether exact numbers were found OR add a suggestion like: "Qualitative mention of [Protein] LC domain - suggest looking up canonical sequence boundaries in UniProt".
5. If no LCRs are found at all, return an empty list.
"""

async def process_pdf_file(pdf_path: str, client: LightLLMClient) -> list[dict]:
    logger.info("Processing file: %s", pdf_path)
    
    raw_text = parse_pdf(pdf_path)
    if not raw_text.strip():
        logger.error("Failed to extract text from PDF: %s", pdf_path)
        return []

    clean_text = prepare_full_text(raw_text)

    # Zapis surowego tekstu z PDF do folderu debug
    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(pdf_path).stem
    
    text_debug_file = debug_dir / f"{pdf_stem}_text.txt"
    with open(text_debug_file, "w", encoding="utf-8") as f:
        f.write(clean_text)
    logger.info("Saved extracted text (%d chars) to: %s", len(clean_text), text_debug_file)

    # Dzielenie na porcje 22k znaków (dopasowane do limitu Groq 8k TPM)
    chunks = chunk_text(clean_text, chunk_size=22000, overlap=3000)
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

        # Pauza 12 sekund, by resetować limit tokenów na minutę
        if idx < len(chunks) - 1:
            await asyncio.sleep(12)

    # Zapis logów diagnostycznych w formacie JSON
    json_debug_file = debug_dir / f"{pdf_stem}_debug.json"
    with open(json_debug_file, "w", encoding="utf-8") as f:
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
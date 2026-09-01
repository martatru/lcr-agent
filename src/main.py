import os
import json
import logging
import asyncio
from pathlib import Path
from pypdf import PdfReader
from llm_client import LightLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT_LCR = """You are a specialized biocuration AI. Your task is to extract all experimentally verified Low Complexity Regions (LCRs) present in the provided text.

Guidelines:
1. Extract every qualifying LCR found in the text. The number of returned items should match the actual findings (it can be one, multiple, or none).
2. Do not invent or force entries; extract only what is explicitly supported by the text.
3. Include the start and end residue positions, proposed function, and exact verbatim evidence.
4. If no LCRs are found, return an empty JSON array: [].
"""

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 10000, overlap: int = 1000) -> list[str]:
    """Splits text into smaller chunks (10k chars) to prevent the model from missing data."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

async def process_pdf_file(pdf_path: str, client: LightLLMClient) -> list[dict]:
    logger.info("Processing file: %s", pdf_path)
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text.strip():
        logger.error("Failed to extract text from PDF: %s", pdf_path)
        return []

    chunks = chunk_text(raw_text, chunk_size=10000, overlap=1000)
    all_annotations = []
    debug_logs = []

    for idx, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d...", idx + 1, len(chunks))
        annotations = await client.generate_lcr_annotations(PROMPT_LCR, chunk)
        
        debug_logs.append({
            "chunk_index": idx,
            "chunk_length": len(chunk),
            "raw_extracted_count": len(annotations),
            "raw_annotations": annotations
        })

        if annotations:
            all_annotations.extend(annotations)

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

    logger.info("Success! Saved %d annotations in %s", len(results), output_file)

if __name__ == "__main__":
    asyncio.run(main())
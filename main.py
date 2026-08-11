import asyncio
import os
import json
import logging
from pathlib import Path

from src.dify_client import DifyClient
from src.text_processor import extract_relevant_sections
from src.llm_client import LightLLMClient
from src.validator import validate_lcr_annotations

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Replace this string with your real API Key from Dify
DIFY_API_KEY = "YOUR_ACTUAL_DIFY_API_KEY_HERE"
LIGHTLLM_URL = "http://localhost:8080" # Update if your LLM server runs elsewhere
RAW_PDF_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "final_results.jsonl"

SYSTEM_PROMPT = """You are a specialized biocuration AI. Your task is to extract ALL experimentally verified Low Complexity Regions (LCRs) with defined sequence boundaries and confirmed biological functions from the provided text.

CRITICAL RULES:
1. EXTRACT ALL MATCHES: Scan the ENTIRE document thoroughly. Do NOT stop after finding the first LCR. Extract every single entry that matches the criteria.
2. ZERO MATCHES HANDLING: If NO qualifying experimentally verified LCRs are found in the text, return an empty JSON array: [].
3. EXPERIMENTAL VALIDATION ONLY: Extract ONLY LCRs confirmed experimentally in the paper. Ignore computational predictions (in silico).
4. DEFINED BOUNDARIES REQUIRED: Explicit numerical start and end residue positions must be extracted as "start_of_annotation" and "end_of_annotation".
5. FUNCTION REQUIRED: Must have an explicitly stated biological or molecular function (e.g., RNA binding, phase separation, transcriptional activation).
6. EXACT EVIDENCE: Provide the exact verbatim sentence from the text as proof for each LCR found.
7. STRICT JSON OUTPUT: Return ONLY a valid JSON array matching the schema below. No markdown wrappers, no preamble, no commentary.

JSON SCHEMA:
[
  {
    "start_of_annotation": integer,
    "end_of_annotation": integer,
    "proposed_function": "string",
    "evidence": "string"
  }
]"""

async def process_pdf(pdf_path: Path, dify_client: DifyClient, llm_client: LightLLMClient, semaphore: asyncio.Semaphore) -> list:
    """Processes a single PDF through the entire pipeline."""
    async with semaphore:
        logger.info(f"Processing {pdf_path.name}...")

        # 1. Extraction (Dify)
        raw_text = await dify_client.extract_text_from_pdf(str(pdf_path))
        if not raw_text:
            logger.error(f"Failed to extract text from {pdf_path.name}")
            return []

        # 2. Filtering Sections
        filtered_text = extract_relevant_sections(raw_text)
        if not filtered_text:
            logger.warning(f"No Results/Discussion sections found in {pdf_path.name}. Using full text fallback.")
            filtered_text = raw_text

        # 3. LLM Inference
        llm_output = await llm_client.generate_lcr_annotations(SYSTEM_PROMPT, filtered_text)

        # 4. Strict Anti-Hallucination Validation
        valid_annotations = validate_lcr_annotations(llm_output, filtered_text)

        # Track source
        for annotation in valid_annotations:
            annotation["source_file"] = pdf_path.name

        logger.info(f"Finished {pdf_path.name}. Found {len(valid_annotations)} valid LCRs.")
        return valid_annotations

async def main():
    # Setup directories
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dify_client = DifyClient(api_key=DIFY_API_KEY)
    llm_client = LightLLMClient(base_url=LIGHTLLM_URL)
    
    # Process max 5 PDFs at the same time to avoid OOM
    semaphore = asyncio.Semaphore(5)

    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {RAW_PDF_DIR}. Add files and run again.")
        return

    logger.info(f"Found {len(pdf_files)} PDFs. Starting pipeline...")

    # Launch concurrent tasks
    tasks = [process_pdf(pdf, dify_client, llm_client, semaphore) for pdf in pdf_files]
    results = await asyncio.gather(*tasks)

    # Flatten results
    all_valid_annotations = [item for sublist in results for item in sublist]

    # Save to JSON Lines
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for annotation in all_valid_annotations:
            f.write(json.dumps(annotation) + "\n")

    logger.info(f"Pipeline complete. Saved {len(all_valid_annotations)} valid annotations to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Core pipeline orchestration module for PDF text extraction and LLM biocuration.
"""

import json
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from llm_client import LightLLMClient
from generate_report import generate_html_report
from text_extractor import parse_pdf, extract_core_results_only, chunk_text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# PROMPT_LCR = """You are an expert biocuration AI. Your task is to exhaustively extract ALL Low Complexity Regions (LCRs/LCDs/IDRs/PLDs) mentioned in the text—both those with explicit residue numbers and those described qualitatively.

# Guidelines:
# 1. Extract every protein mentioned to contain or form an LCR/LCD (e.g., FUS, TIA1, hnRNPA1, hnRNPA2, CIRBP, RBM3, Sup35, TDP43, FMRP).
# 2. If numerical residue coordinates are given (e.g., 'residues 2-214'), extract them as strings in start_of_annotation and end_of_annotation. If no numbers are provided in the text, set them to 'Unspecified'.
# 3. 'evidence' MUST be an exact verbatim sentence from the text proving the LCR and its position/function.
# 4. In 'curator_note', state whether exact numbers were found OR add a suggestion like: "Qualitative mention of [Protein] LC domain - suggest looking up canonical sequence boundaries in UniProt".
# 5. If no LCRs are found at all, return an empty list.
# """

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

async def process_pdf_file(pdf_path: str, client: LightLLMClient) -> list[dict]:
    """Processes a single PDF file, chunks text, queries Groq API, and validates results."""
    logger.info("Processing file: %s", pdf_path)

    raw_text = parse_pdf(pdf_path)
    if not raw_text.strip():
        logger.error("Failed to extract text from PDF: %s", pdf_path)
        return []

    clean_text = extract_core_results_only(raw_text)

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
        
        all_annotations.extend(annotations)

        debug_logs.append({
            "chunk_index": idx,
            "chunk_length": len(chunk),
            "raw_extracted_count": len(annotations),
            "raw_annotations": annotations,
        })

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

    logger.info("Generating HTML biocuration report...")
    generate_html_report(
        input_file=str(jsonl_output_file),
        output_html=str(html_report_file)
    )

    logger.info("Pipeline finished! View report at: %s", html_report_file)


if __name__ == "__main__":
    asyncio.run(main())
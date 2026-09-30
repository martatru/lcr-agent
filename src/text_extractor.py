"""
Unified Text Extraction and Processing Module for Scientific Literature.
"""

import logging
import re
import unicodedata
from typing import List, Dict, Any, Optional

import pymupdf

logger = logging.getLogger(__name__)

CAPTION_PATTERN = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table|Scheme)\s*\d+[\.:\s]", re.IGNORECASE
)

# Standard pattern matching DOIs (e.g., 10.1038/s41594-021-00624-8)
DOI_REGEX = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b",
    re.IGNORECASE
)


def _clean_unicode_and_ocr_artifacts(text: str) -> str:
    """
    Normalize Unicode characters and repair OCR artifacts while preserving mutations and figure panels.
    """
    # 1. NFKC Normalization
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Repair °C degrees ONLY within explicit temperature contexts
    temp_context_pattern = re.compile(
        r"(?i)\b(at|incubated|temperature|temp\.?|heat(?:ed)?|warm(?:ed)?|cool(?:ed)?|ambient)\s*(\d+(?:\.\d+)?)\s*°?\s*C\b"
    )
    normalized = temp_context_pattern.sub(r"\1 \2°C", normalized)

    # Clean up trailing spaces in explicit degree patterns like "37 ° C" -> "37°C"
    normalized = re.sub(r"(\d+)\s*°\s*C\b", r"\1°C", normalized)
    
    # 3. Repair micro-molar concentration units (uM -> μM)
    normalized = re.sub(r"(\b\d+(?:\.\d+)?)\s*(?:uM|lM)\b", r"\1 μM", normalized)

    # Replace unprintable Private Use Area (PUA) glyphs
    normalized = re.sub(r"[\uE000-\uF8FF]", "\ufffd", normalized)

    return normalized


def extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract the first valid DOI string found in the text using regex.
    Cleans up trailing punctuation attached by text boundaries.
    """
    matches = DOI_REGEX.findall(text)
    for match in matches:
        clean_doi = re.sub(r"[\.:;,()]+$", "", match).strip()
        if len(clean_doi) > 7:
            return clean_doi
    return None


def extract_tagged_sections(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extract text blocks from a PDF categorized by tagged sections:
    ABSTRACT, RESULTS, METHODS, CAPTION, BODY.
    """
    doc = pymupdf.open(pdf_path)
    sections: List[Dict[str, str]] = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for b in blocks:
            block_text = b[4].strip()
            if not block_text:
                continue

            block_text = _clean_unicode_and_ocr_artifacts(block_text)

            # Section identification
            if CAPTION_PATTERN.match(block_text):
                section_type = "CAPTION"
            elif re.search(r"(?i)^\s*(abstract|summary)\b", block_text):
                section_type = "ABSTRACT"
            elif re.search(r"(?i)^\s*(methods|experimental procedures|materials and methods)\b", block_text):
                section_type = "METHODS"
            elif re.search(r"(?i)^\s*(results|discussion)\b", block_text):
                section_type = "RESULTS"
            else:
                section_type = "BODY"

            sections.append({
                "page": page_num,
                "section": section_type,
                "text": block_text
            })

    return sections


def chunk_sections_by_paragraphs(
    sections: List[Dict[str, str]], 
    max_chunk_size: int = 8000
) -> List[str]:
    """
    Group paragraphs into chunks respecting token limits and section boundaries.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for sec in sections:
        header = f"[SECTION: {sec['section']}]\n"
        paragraph = header + sec['text'] + "\n\n"
        
        if current_length + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append("".join(current_chunk).strip())
            current_chunk = [paragraph]
            current_length = len(paragraph)
        else:
            current_chunk.append(paragraph)
            current_length += len(paragraph)

    if current_chunk:
        chunks.append("".join(current_chunk).strip())

    return chunks


def deduplicate_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate annotations resulting from overlapping text chunks."""
    unique = []
    seen = set()

    for ann in annotations:
        key = (
            ann.get("protein_name", "").lower(),
            ann.get("start_of_annotation"),
            ann.get("end_of_annotation"),
            ann.get("binding_target"),
            ann.get("evidence", "").strip().lower()
        )
        if key not in seen:
            seen.add(key)
            unique.append(ann)

    return unique
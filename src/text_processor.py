"""
Text Processing and Section Extraction Module for Scientific Literature.

Splits parsed PDF text into structured sections using header detection,
retaining only Results, Discussion, Conclusions, and related key sections.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Whitelisted section headers to retain for LLM analysis
WHITELIST_HEADERS = [
    r"results",
    r"discussion",
    r"results\s+and\s+discussion",
    r"conclusions?",
    r"concluding\s+remarks",
    r"summary(?:\s+and\s+outlook)?",
    r"perspectives?",
    r"key\s+findings",
]

# Blacklisted section headers to discard
BLACKLIST_HEADERS = [
    r"abstract",
    r"introduction",
    r"background",
    r"(?:materials?\s+and\s+)?methods",
    r"experimental\s+procedures",
    r"methodology",
    r"references",
    r"bibliography",
    r"literature\s+cited",
    r"acknowledg?ment[s]?",
    r"author\s+contributions?",
    r"competing\s+interests?",
    r"data\s+availability",
]


def prepare_full_text(text: str) -> str:
    """Clean basic whitespace artifacts from PDF text extraction."""
    clean = re.sub(r"\r\n|\r", "\n", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = re.sub(r"[ \t]+", " ", clean)
    return clean.strip()


def _is_matching(header_text: str, patterns: List[str]) -> bool:
    """Check if header text matches any regex pattern in a list."""
    header_text = header_text.strip().lower()
    for pattern in patterns:
        if re.search(r"\b" + pattern + r"\b", header_text):
            return True
    return False


def _fallback_extraction(text: str) -> str:
    """Fallback heuristic stripping references and early sections when headers fail."""
    ref_match = re.search(
        r"(?i)\n\s*(references|bibliography|literature\s+cited)\b", text
    )
    if ref_match:
        text = text[: ref_match.start()]

    results_match = re.search(
        r"(?i)\n\s*(results|results\s+and\s+discussion)\b", text
    )
    if results_match:
        text = text[results_match.start() :]
    else:
        skip_len = int(len(text) * 0.20)
        text = text[skip_len:]

    return text.strip()


def extract_core_results_only(text: str) -> str:
    """
    Extract Results, Discussion, Conclusions, and Summary sections from text.

    Parses document header-to-header, filtering out Abstract, Introduction,
    Methods, and Reference blocks regardless of section order.
    """
    cleaned_text = prepare_full_text(text)
    if not cleaned_text:
        return ""

    # Regex matching section titles at the start of lines (with optional numbering)
    header_regex = re.compile(
        r"(?m)^\s*(?:[0-9]{1,2}(?:\.[0-9]{1,2})*\.?)?\s*"
        r"([A-Z][A-Za-z0-9\s,\-&]{2,60})\s*$"
    )

    matches = list(header_regex.finditer(cleaned_text))

    if not matches:
        logger.warning(
            "No explicit section headers found. Falling back to heuristic slicing."
        )
        return _fallback_extraction(cleaned_text)

    blocks: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        header_name = match.group(0).strip()
        start_pos = match.end()
        end_pos = (
            matches[idx + 1].start()
            if idx + 1 < len(matches)
            else len(cleaned_text)
        )
        content = cleaned_text[start_pos:end_pos].strip()
        blocks.append((header_name, content))

    selected_texts: List[str] = []
    current_state = "UNKNOWN"

    for header_name, content in blocks:
        if _is_matching(header_name, BLACKLIST_HEADERS):
            current_state = "BLACKLISTED"
            continue

        if _is_matching(header_name, WHITELIST_HEADERS):
            current_state = "WHITELISTED"
            selected_texts.append(f"--- {header_name} ---\n{content}")
            continue

        # If block is a sub-header inside a whitelisted parent section
        if current_state == "WHITELISTED":
            selected_texts.append(f"\n{header_name}\n{content}")

    result_text = "\n\n".join(selected_texts).strip()

    # Trigger fallback if block extraction returned negligible text
    if len(result_text) < 300:
        logger.info(
            "Extracted section too small (%d chars). Using fallback.",
            len(result_text),
        )
        return _fallback_extraction(cleaned_text)

    logger.info(
        "Retained target sections (%d -> %d chars).",
        len(text),
        len(result_text),
    )
    return result_text


def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 1000) -> List[str]:
    """Split text into overlapping chunks for downstream LLM inference."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
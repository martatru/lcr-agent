"""
Unified Text Extraction and Processing Module for Scientific Literature.

Combines zero-OCR PDF parsing (PyMuPDF) with bounding-box filtering,
Unicode normalization, layout-aware block sorting, de-hyphenation,
caption removal, OCR artifact repairs (including °C mistranslations and scientific units),
and header whitelist/blacklist section segmentation.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import pymupdf

logger = logging.getLogger(__name__)

# Whitelisted section headers to retain for downstream analysis
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

# Blacklisted section headers to explicitly discard
BLACKLIST_HEADERS = [
    r"abstract",
    r"introduction",
    r"background",
    r"(?:materials?\s+and\s+)?methods",
    r"experimental\s+(?:procedures|section)",
    r"star\+?\s*methods",
    r"method\s+details",
    r"methodology",
    r"references",
    r"bibliography",
    r"literature\s+cited",
    r"acknowledg?ment[s]?",
    r"author\s+contributions?",
    r"competing\s+interests?",
    r"data\s+availability",
    r"supplementary\s+information",
    r"reporting\s+summary",
]

# Patterns for figure/table/scheme captions to exclude
CAPTION_PATTERN = re.compile(
    r"^\s*(?:Figure|Fig\.|Table|Scheme)\s+\d+[\.:\s]", re.IGNORECASE
)

# Recurring journal header/footer noise patterns
HEADER_FOOTER_PATTERNS = [
    r"downloaded\s+from",
    r"academic\.oup\.com",
    r"molecular\s+cell",
    r"journal\s+of",
    r"nucleic\s+acids\s+research",
    r"http[s]?://",
    r"doi\.org",
    r"reporting\s+summary",
]


def _clean_unicode_and_ocr_artifacts(text: str) -> str:
    """
    Normalize Unicode characters and repair common OCR/parsing artifacts:
    - Resolves split diacritics / umlauts and ligatures using NFKC.
    - Fixes accidental replacement of 'C' with '°C' in figure references, protein/domain names, and mutations.
    - Repairs misread scientific units and degrees.
    """
    # 1. Normalize Unicode (combines split accents like 'Bru ̈ ckner' -> 'Brückner' and converts ligatures)
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Repair °C mistranslations (where letter 'C' was mistakenly converted to '°C')
    # Figure references: Figure 1°C -> Figure 1C, Fig. 2°C -> Fig. 2C
    normalized = re.sub(r"(?i)\b(Fig(?:ure)?\s*\d+)\s*°C\b", r"\1C", normalized)

    # C-terminal domains: °C-terminal -> C-terminal
    normalized = re.sub(r"(?i)°C-terminal\b", "C-terminal", normalized)

    # Protein/domain names & mutations (e.g., SEC24°C -> SEC24C, R391°C -> R391C)
    # Using group capture to avoid Python variable-width look-behind limitation
    normalized = re.sub(r"\b([A-Za-z0-9]+)°C\b", r"\1C", normalized)

    # 3. Scientific unit & symbol repairs
    normalized = re.sub(r"(\d+)\s*[\uE000-\uF8FF]?\s*°?\s*C\b", r"\1°C", normalized)
    normalized = re.sub(r"(\b\d+(?:\.\d+)?)\s*(?:uM|lM)\b", r"\1 μM", normalized)

    # Strip remaining Private Use Area (PUA) Unicode characters
    normalized = re.sub(r"[\uE000-\uF8FF]", "", normalized)

    return normalized


def _dehyphenate(text: str) -> str:
    """Reconstruct words split across linebreaks (e.g., 'se-\n vere' -> 'severe')."""
    return re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)


def parse_pdf(pdf_path: str) -> str:
    """
    Extract clean text from PDF using PyMuPDF with layout-aware block parsing,
    bounding-box filtering (top/bottom margins), caption removal, and de-hyphenation.
    """
    try:
        doc = pymupdf.open(pdf_path)
        extracted_blocks: List[str] = []

        for page in doc:
            rect = page.rect
            page_height = rect.height
            page_width = rect.width

            # Define 5% top and bottom margins for header/footer removal
            top_margin = page_height * 0.05
            bottom_margin = page_height * 0.95

            # Get text blocks: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            valid_blocks = []

            for b in blocks:
                if len(b) < 7 or b[6] != 0:  # b[6] == 0 indicates text block
                    continue

                x0, y0, x1, y1, block_text, _, _ = b[:7]

                # Filter out header/footer margin regions
                if y0 < top_margin or y1 > bottom_margin:
                    continue

                stripped_text = block_text.strip()
                if not stripped_text:
                    continue

                # Exclude figure and table legend captions
                if CAPTION_PATTERN.search(stripped_text):
                    continue

                # Exclude journal meta headers and footers
                lower_text = stripped_text.lower()
                if any(re.search(pattern, lower_text) for pattern in HEADER_FOOTER_PATTERNS):
                    continue

                valid_blocks.append((x0, y0, x1, y1, block_text))

            # Sort blocks to handle standard two-column layout properly
            mid_x = page_width / 2.0
            valid_blocks.sort(key=lambda block: (block[0] >= mid_x, block[1]))

            for block in valid_blocks:
                extracted_blocks.append(block[4])

        full_text = "\n\n".join(extracted_blocks)
        full_text = _clean_unicode_and_ocr_artifacts(full_text)
        full_text = _dehyphenate(full_text)
        return full_text

    except Exception as error:
        logger.error("Error reading PDF file %s: %s", pdf_path, error)
        return ""


def prepare_full_text(text: str) -> str:
    """Clean basic whitespace artifacts from PDF text extraction."""
    clean = re.sub(r"\r\n|\r", "\n", text)
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def _is_matching(header_text: str, patterns: List[str]) -> bool:
    """Check if header text matches any regex pattern in the given list."""
    header_text = header_text.strip().lower()
    header_text = re.sub(r"^[\-\s=]+|[\-\s=]+$", "", header_text)
    for pattern in patterns:
        if re.search(r"\b" + pattern + r"\b", header_text):
            return True
    return False


def _fallback_extraction(text: str) -> str:
    """Fallback heuristic slicing references, method sections, and introductory sections when headers fail."""
    ref_match = re.search(
        r"(?i)\n\s*(?:---?\s*)?(references|bibliography|literature\s+cited|experimental\s+(?:procedures|section)|star\+?\s*methods|method\s+details|reporting\s+summary)\b",
        text,
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
    Methods, STAR+Methods, Reporting Summary, and Reference blocks regardless of section order.
    """
    cleaned_text = prepare_full_text(text)
    if not cleaned_text:
        return ""

    header_regex = re.compile(
        r"(?m)^\s*(?:[\-\s=]*)(?:[0-9]{1,2}(?:\.[0-9]{1,2})*\.?)?\s*"
        r"([A-Z][A-Za-z0-9\s,\-&]{2,60})\s*(?:[\-\s=]*)$"
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

        if current_state == "WHITELISTED":
            selected_texts.append(f"\n{header_name}\n{content}")

    result_text = "\n\n".join(selected_texts).strip()

    # Hard truncation cutoff for trailing administrative / methodology sections
    result_text = re.sub(
        r"(?i)\n\s*---?\s*(?:star\+?\s*methods|method\s+details|experimental\s+(?:procedures|section)|reporting\s+summary|references|bibliography)\b.*$",
        "",
        result_text,
        flags=re.DOTALL,
    ).strip()

    if len(result_text) < 300:
        logger.info(
            "Extracted section too small (%d chars). Using fallback extraction.",
            len(result_text),
        )
        return _fallback_extraction(cleaned_text)

    logger.info(
        "Retained target sections (%d -> %d chars).",
        len(text),
        len(result_text),
    )
    return result_text


def chunk_text(text: str, chunk_size: int = 12000, overlap: int = 2000) -> List[str]:
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


def process_pdf_text(
    pdf_path: str, chunk_size: int = 12000, overlap: int = 2000
) -> Tuple[str, List[str]]:
    """
    High-level convenience function: parses PDF, extracts core target sections,
    and returns both filtered full text and chunked text.
    """
    raw_text = parse_pdf(pdf_path)
    if not raw_text.strip():
        return "", []

    filtered_text = extract_core_results_only(raw_text)
    chunks = chunk_text(filtered_text, chunk_size=chunk_size, overlap=overlap)
    return filtered_text, chunks
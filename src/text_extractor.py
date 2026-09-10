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
    r"(?:quantification\s+and\s+)?statistical\s+analysis", 
    r"references",
    r"bibliography",
    r"literature\s+cited",
    r"acknowledg?ment[s]?(?:\s+and\s+funding)?",           
    r"funding",                                             
    r"author\s+contributions?",
    r"competing\s+interests?",
    r"data(?:\s+and\s+code)?\s+availability",               
    r"supplementar(?:y|ial)\s+information",
    r"supplemental\s+information",                          
    r"supporting\s+information",
    r"associated\s+content",
    r"author\s+information",
    r"reporting\s+summary",
    r"significance"
]

# Bullet/dingbat glyphs some journals (notably ACS titles) prepend to their
# top-level section headers instead of relying on font size alone, e.g.
# "\u25a0RESULTS", "\u25a0EXPERIMENTAL SECTION", "\u25a0ACKNOWLEDGMENTS".
# These are not ordinary punctuation, so they must be handled explicitly
# wherever a header line is recognized or cleaned up.
_HEADER_BULLET_CHARS = "\u25a0\u25aa\u25cf\u25c6\u2666\u2022\u25a1"

# Some journals typeset subsection headers "run-in": fused onto the same
# physical line as the body text that follows, joined by a dash
# (e.g. "Acknowledgments\u2014We thank Jack Werren ..."). `header_regex` in
# `extract_core_results_only` only recognizes a header that occupies a whole
# line by itself, so it never fires on these - which is exactly what's
# wanted for run-in *whitelisted* subheadings (they're legitimate
# Results/Discussion body content, e.g. "Low-complexity Domains Mediate
# Transcriptional Activation by ZLD\u2014The activation domain..."). But it
# means a run-in *blacklisted* header (Acknowledgments, Author
# Contributions, Data Availability, etc.) - which conventionally sits at the
# very end of a paper, after Discussion/Conclusions - silently evades
# blacklisting and leaks its whole body into the extracted output. This
# pattern catches that case specifically so it can be truncated away.
RUNIN_BLACKLIST_HEADER_PATTERN = re.compile(
    r"(?m)^[ \t]*[" + _HEADER_BULLET_CHARS + r"]?[ \t]*(?:"
    + "|".join(BLACKLIST_HEADERS)
    + r")[ \t]*[\u2014\u2013-][ \t]*\S",
    re.IGNORECASE,
)

# Patterns for figure/table/scheme captions to exclude.
# `\s*` (rather than `\s+`) between the label and the number is deliberate
# defense in depth: it still matches if a word-joining glitch elsewhere in
# the pipeline ever glues them together (e.g. "Figure3."), which previously
# let whole figure captions leak into the extracted Results/Discussion text.
CAPTION_PATTERN = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table|Scheme)\s*\d+[\.:\s]", re.IGNORECASE
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


# Sentinel used to temporarily mask a "C" whose spurious leading degree sign
# has just been removed (see `_clean_unicode_and_ocr_artifacts`), so that the
# temperature/units repair further down the same function cannot immediately
# reinsert the degree sign it just stripped. U+0000 never occurs in text
# extracted from a PDF, so it is safe to use as an internal-only marker.
_PROTECTED_C = "\x00"


def _clean_unicode_and_ocr_artifacts(text: str) -> str:
    """
    Normalize Unicode characters and repair common OCR/parsing artifacts:
    - Resolves split diacritics / umlauts and ligatures using NFKC.
    - Fixes accidental replacement of 'C' with '°C' in figure references,
      protein/domain names, and mutations.
    - Repairs misread scientific units and degrees.
    """
    # 1. Normalize Unicode (combines split accents like 'Bru \u0308 ckner' ->
    # 'Brückner' and converts ligatures)
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Repair spurious degree signs that PDF extraction inserts before a
    # bare "C" in figure references, C-terminal domain names, and
    # protein/mutation names (e.g. "Fig. 1°C" -> "Fig. 1C", "SEC24°C" ->
    # "SEC24C", "R391°C" -> "R391C").
    #
    # Every fix below writes _PROTECTED_C instead of a literal "C". This
    # matters: step 3 re-adds a missing degree sign to any "<digits>C" it
    # finds (e.g. "37C" -> "37°C"), and "Fig. 1°C" -> "Fig. 1C" produced by
    # this step has exactly that shape, so without the mask step 3 would
    # instantly undo the fix just made here. The mask is only turned back
    # into "C" once step 3 has already run.

    # Figure references: "Figure 1°C" / "Fig. 2°C" -> "Fig. 2" + masked C.
    # `\.?` accounts for the abbreviation's period ("Fig."), which the
    # original pattern omitted and which caused it to never match.
    normalized = re.sub(
        r"(?i)\b(Fig(?:ure)?\.?\s*\d+)\s*°C\b", rf"\1{_PROTECTED_C}", normalized
    )

    # C-terminal domains: °C-terminal -> C-terminal
    normalized = re.sub(r"(?i)°C-terminal\b", f"{_PROTECTED_C}-terminal", normalized)

    # Protein/domain names & mutations (e.g., SEC24°C -> SEC24C, R391°C -> R391C).
    # The captured run must contain a letter, not just digits: a bare numeral
    # like "37°C" is a genuine, already-correct temperature reading (left
    # untouched by step 3 below), not an OCR artifact to repair, and must not
    # be stripped of its degree sign here.
    normalized = re.sub(
        r"\b((?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]+)°C\b",
        rf"\1{_PROTECTED_C}",
        normalized,
    )

    # 3. Scientific unit & symbol repairs (adds a missing degree sign for
    # genuine temperature values, e.g. "37C" -> "37°C"). Occurrences masked
    # above are immune since they no longer contain a literal "C" here.
    normalized = re.sub(r"(\d+)\s*[\uE000-\uF8FF]?\s*°?\s*C\b", r"\1°C", normalized)
    normalized = re.sub(r"(\b\d+(?:\.\d+)?)\s*(?:uM|lM)\b", r"\1 μM", normalized)

    # Any remaining Private Use Area (PUA) codepoint is an embedded-font
    # glyph (often a symbol such as "=" or "±") that PyMuPDF could not map
    # to a real Unicode character. These are replaced with U+FFFD ("<20>")
    # rather than deleted outright: silently dropping them turns e.g. "n = 3,
    # mean +/- S.D." into "n  3, mean  S.D.", quietly destroying scientific
    # meaning. Leaving a visible placeholder makes the loss searchable
    # instead of silent; recovering the actual symbol requires inspecting
    # the specific source PDF's embedded font glyph map, which is outside
    # what can be done generically here. Then unmask the "C"s from step 2.
    normalized = re.sub(r"[\uE000-\uF8FF]", "\ufffd", normalized)
    normalized = normalized.replace(_PROTECTED_C, "C")

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

            # Get words: (x0, y0, x1, y1, word, block_no, line_no, word_no)
            words = page.get_text("words")

            # Group words by their source block/line so each block's text
            # can be rebuilt line by line, in original reading order.
            blocks_by_no: dict = {}
            for x0, y0, x1, y1, word, block_no, line_no, _word_no in words:
                block = blocks_by_no.setdefault(block_no, {})
                block.setdefault(line_no, []).append((x0, y0, x1, y1, word))

            valid_blocks = []
            for block_lines in blocks_by_no.values():
                all_words = [w for line in block_lines.values() for w in line]
                if not all_words:
                    continue

                x0 = min(w[0] for w in all_words)
                y0 = min(w[1] for w in all_words)
                x1 = max(w[2] for w in all_words)
                y1 = max(w[3] for w in all_words)

                # Filter out header/footer margin regions
                if y0 < top_margin or y1 > bottom_margin:
                    continue

                line_texts = []
                for line_no in sorted(block_lines):
                    line_words = sorted(block_lines[line_no], key=lambda w: w[0])
                    line_text = " ".join(w[4] for w in line_words).strip()
                    if not line_text:
                        continue

                    lower_line = line_text.lower()

                    is_strict_noise = any(
                        re.search(pattern, lower_line) for pattern in [
                            r"downloaded\s+from",
                            r"academic\.oup\.com",
                            r"http[s]?://",
                            r"doi\.org",
                            r"reporting\s+summary",
                        ]
                    )

                    is_short_generic_noise = (len(line_text.split()) <= 12) and any(
                        re.search(pattern, lower_line) for pattern in [
                            r"molecular\s+cell",
                            r"journal\s+of",
                            r"nucleic\s+acids\s+research",
                        ]
                    )

                    if is_strict_noise or is_short_generic_noise:
                        continue

                    line_texts.append(line_text)

                block_text = "\n".join(line_texts).strip()

                if not block_text:
                    continue

                # Exclude figure and table legend captions
                if CAPTION_PATTERN.search(block_text):
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
    strip_chars = r"\-\s=" + _HEADER_BULLET_CHARS
    header_text = re.sub(
        r"^[" + strip_chars + r"]+|[" + strip_chars + r"]+$", "", header_text
    )
    for pattern in patterns:
        if re.search(r"\b" + pattern + r"\b", header_text):
            return True
    return False


# Connector words that are conventionally left lowercase inside a Title
# Case heading (e.g. "Results and Discussion", "Conservation of ZLD and
# TAGteam Sites in Driving Genome Activation").
_HEADER_CONNECTOR_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "via", "with",
}


def _looks_like_header(candidate: str) -> bool:
    candidate = candidate.strip()
    if not candidate or candidate.endswith(","):
        return False

    # Usunięcie końcowej interpunkcji przed ścisłym dopasowaniem
    clean_candidate = re.sub(r"[\.:/]$", "", candidate).strip()
    lowered = clean_candidate.lower()

    if any(
        re.fullmatch(pattern, lowered)
        for pattern in WHITELIST_HEADERS + BLACKLIST_HEADERS
    ):
        return True

    words = clean_candidate.split()
    if not words:
        return False

    if clean_candidate.replace(" ", "").isupper():
        return True

    for word in words:
        bare = word.strip(",&-")
        if not bare:
            continue
        if bare.lower() in _HEADER_CONNECTOR_WORDS:
            continue
        if not bare[0].isupper():
            return False
    return True

def _fallback_extraction(text: str) -> str:
    """Fallback heuristic slicing references, method sections, and
    introductory sections when headers fail."""
    trailing_pattern = "|".join([
        r"references", r"bibliography", r"literature\s+cited",
        r"experimental\s+(?:procedures|section)", r"star\+?\s*methods",
        r"method\s+details", r"reporting\s+summary",
        r"acknowledg?ment[s]?(?:\s+and\s+funding)?", r"funding", 
        r"data(?:\s+and\s+code)?\s+availability",
        r"(?:quantification\s+and\s+)?statistical\s+analysis",
        r"supplementar(?:y|ial|al)\s+information", r"supporting\s+information",
        r"author\s+contributions?", r"competing\s+interests?"
    ])
    
    ref_match = re.search(
        rf"(?i)(?:^|\n)\s*(?:---?\s*)?(?:[{_HEADER_BULLET_CHARS}]\s*)?({trailing_pattern})\b",
        text,
    )
    if ref_match:
        text = text[: ref_match.start()]

    results_match = re.search(
        r"(?i)(?:^|\n)\s*(?:[" + _HEADER_BULLET_CHARS + r"]\s*)?"
        r"(results|results\s+and\s+discussion)\b",
        text,
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
        r"(?m)^[ \t]*(?:[\-\t =" + _HEADER_BULLET_CHARS + r"]*)"
        r"(?:[0-9]{1,2}(?:\.[0-9]{1,2})*\.?)?[ \t]*"
        r"([A-Z][A-Za-z0-9 ,\-&:\./]{2,60})[ \t]*(?:[\-\t =]*)$"
    )

    matches = [
        m for m in header_regex.finditer(cleaned_text)
        if _looks_like_header(m.group(1))
    ]

    if not matches:
        logger.warning(
            "No explicit section headers found. Falling back to heuristic slicing."
        )
        return _fallback_extraction(cleaned_text)

    blocks: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        header_name = match.group(0).strip()
        header_name = header_name.lstrip(_HEADER_BULLET_CHARS + " \t")
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

    trailing_pattern = "|".join([
        r"star\+?\s*methods", r"method\s+details", r"experimental\s+(?:procedures|section)",
        r"(?:materials?\s+and\s+)?methods", r"reporting\s+summary",
        r"references", r"bibliography", r"literature\s+cited",
        r"acknowledg?ment[s]?(?:\s+and\s+funding)?", r"funding",
        r"author\s+contributions?", r"competing\s+interests?",
        r"data(?:\s+and\s+code)?\s+availability",
        r"supplementar(?:y|ial|al)\s+information", r"supporting\s+information",
        r"(?:quantification\s+and\s+)?statistical\s+analysis",
    ])
    
    result_text = re.sub(
        rf"(?i)\n\s*(?:---?\s*)?(?:{trailing_pattern})[ \t]*[:\.]?[ \t]*\n.*$",
        "",
        result_text,
        flags=re.DOTALL,
    ).strip()

    # Second cutoff for blacklisted sections typeset "run-in"
    runin_match = RUNIN_BLACKLIST_HEADER_PATTERN.search(result_text)
    if runin_match:
        result_text = result_text[: runin_match.start()].rstrip()

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
import re
import logging

logger = logging.getLogger(__name__)


def prepare_full_text(text: str) -> str:
    """Cleans up basic whitespace artifacts from PDF extraction."""
    clean = re.sub(r'\r\n|\r', '\n', text)
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    clean = re.sub(r'[ \t]+', ' ', clean)
    return clean.strip()


def extract_core_results_only(text: str) -> str:
    """
    Strips Abstract, Introduction, Materials & Methods, Acknowledgments, and References.
    Retains ONLY Results and Discussion sections to maximize token efficiency.
    """
    # 1. Strip References, Bibliography, Acknowledgments from the end
    end_patterns = [
        r"(?i)\n\s*(references|bibliography|literature\s+cited)\b",
        r"(?i)\n\s*(acknowledgments|acknowledgements|author\s+contributions)\b"
    ]
    for pattern in end_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]

    # 2. Identify section headers
    results_match = re.search(r"(?i)\n\s*(results|results\s+and\s+discussion)\b", text)
    methods_match = re.search(r"(?i)\n\s*(materials\s+and\s+methods|experimental\s+procedures|methods|methodology)\b", text)
    intro_match = re.search(r"(?i)\n\s*(introduction|background)\b", text)

    if results_match:
        results_start = results_match.start()
        # If Methods is placed after Results (e.g. Nature/Science format), chop before Methods
        if methods_match and methods_match.start() > results_start:
            core_text = text[results_start:methods_match.start()]
        else:
            core_text = text[results_start:]
    else:
        # Fallback if explicit "Results" header is missing
        if methods_match:
            core_text = text[methods_match.end():]
        elif intro_match:
            core_text = text[intro_match.start() + 3000:]
        else:
            skip_len = int(len(text) * 0.25)  # Skip first 25% (abstract/intro)
            core_text = text[skip_len:]

    # Secondary check to remove any remaining Methods block
    m_check = re.search(r"(?i)\n\s*(materials\s+and\s+methods|experimental\s+procedures|methods)\b", core_text)
    if m_check:
        core_text = core_text[:m_check.start()]

    logger.info("Retained only Results & Discussion (%d -> %d chars).", len(text), len(core_text))
    return core_text.strip()


def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 1000) -> list[str]:
    """Splits text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks
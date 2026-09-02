import re
import logging
import unicodedata

logger = logging.getLogger(__name__)


def fix_hyphenation(text: str) -> str:
    """Fixes words split across line breaks with a hyphen (e.g. 'con-\\ntinuously' -> 'continuously')."""
    return re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', text)


def clean_whitespace_and_unicode(text: str) -> str:
    """Normalizes unicode characters, removes control characters, and cleans excessive whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'[\r\t\f\v]', ' ', text)
    text = re.sub(r'[ ]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_references_section(raw_text: str) -> str:
    """Cuts off the References / Bibliography section from the end of the text."""
    patterns = [
        r'(?im)^\s*(?:#+\s*)?(?:\d+[\.\s]*)?(?:\*\*)?\b(?:References|Bibliography|Literature Cited|References and Notes)\b(?:\*\*)?\s*$',
    ]

    clean_text = raw_text
    for pattern in patterns:
        matches = list(re.finditer(pattern, clean_text))
        if matches:
            for m in reversed(matches):
                if m.start() > len(clean_text) * 0.4:
                    clean_text = clean_text[:m.start()]
                    logger.info("Cut off references section starting at character index %d", m.start())
                    break
            break

    return clean_text.strip()


def prepare_full_text(raw_text: str) -> str:
    """Main pipeline for cleaning and normalizing PDF text before sending to LLM."""
    if not raw_text:
        return ""

    text = remove_references_section(raw_text)
    text = fix_hyphenation(text)
    text = clean_whitespace_and_unicode(text)

    return text


# Alias for backward compatibility
extract_relevant_sections = prepare_full_text


def chunk_text(text: str, chunk_size: int = 12000, overlap: int = 2000) -> list[str]:
    """Splits text into overlapping chunks, attempting to break on paragraph or sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:])
            break

        split_pos = text.rfind('\n\n', start + chunk_size // 2, end)
        if split_pos == -1:
            split_pos = text.rfind('\n', start + chunk_size // 2, end)
        if split_pos == -1:
            split_pos = text.rfind('. ', start + chunk_size // 2, end)
            if split_pos != -1:
                split_pos += 1

        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunks.append(text[start:split_pos].strip())
        start = max(split_pos - overlap, start + 1)

    return chunks
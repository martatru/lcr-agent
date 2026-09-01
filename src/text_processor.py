import re
import logging
import unicodedata

logger = logging.getLogger(__name__)


def fix_hyphenation(text: str) -> str:
    """
    Fixes words split across line breaks with a hyphen (e.g. 'con-\ntinuously' -> 'continuously').
    Preserves numerical ranges like '10-\n20' or 'aa 150-\n180'.
    """
    # Fix word-hyphen-newline-word pattern
    text = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
    return text


def clean_whitespace_and_unicode(text: str) -> str:
    """
    Normalizes unicode characters (ligatures like \ufb01 -> fi),
    replaces non-breaking spaces, and cleans excessive whitespace.
    """
    # Normalize unicode ligatures and special chars
    text = unicodedata.normalize("NFKC", text)
    # Replace non-breaking spaces and other special space chars with standard space
    text = re.sub(r'[\r\t\f\v]', ' ', text)
    # Merge multiple spaces on the same line into single space
    text = re.sub(r'[ ]+', ' ', text)
    # Merge more than two consecutive newlines into double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_references_section(raw_text: str) -> str:
    """
    Cuts off the References / Bibliography section from the end of the paper
    to avoid sending non-essential text to the LLM.
    """
    patterns = [
        r'(?im)^\s*(?:#+\s*)?(?:\d+[\.\s]*)?(?:\*\*)?\b(?:References|Bibliography|Literature Cited|References and Notes)\b(?:\*\*)?\s*$',
    ]
    
    clean_text = raw_text
    for pattern in patterns:
        matches = list(re.finditer(pattern, clean_text))
        if matches:
            # Pick the last match that occurs after at least 40% of the text length
            for m in reversed(matches):
                if m.start() > len(clean_text) * 0.4:
                    clean_text = clean_text[:m.start()]
                    logger.info("Cut off references section starting at character index %d", m.start())
                    break
            break
            
    return clean_text.strip()


def prepare_full_text(raw_text: str) -> str:
    """
    Main pipeline for text cleaning and normalization before passing to LLM.
    """
    if not raw_text:
        return ""
    
    # 1. Strip references section from paper
    text = remove_references_section(raw_text)
    
    # 2. Fix split words (hyphenation at line breaks)
    text = fix_hyphenation(text)
    
    # 3. Clean spaces and unicode ligatures
    text = clean_whitespace_and_unicode(text)
    
    return text


# Alias for backwards compatibility
extract_relevant_sections = prepare_full_text


def chunk_text(text: str, chunk_size: int = 80000, overlap: int = 5000) -> list[str]:
    """
    Splits long text into overlapping chunks, attempting to split on paragraph or sentence boundaries
    rather than cutting arbitrarily in the middle of words.
    
    Default chunk_size is 80,000 chars (~20k tokens), well suited for models with large context windows (e.g. Groq 128k).
    """
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

        # Try to find a paragraph break or newline near the end of the chunk
        split_pos = text.rfind('\n\n', start + chunk_size // 2, end)
        if split_pos == -1:
            # Fallback to single newline
            split_pos = text.rfind('\n', start + chunk_size // 2, end)
        if split_pos == -1:
            # Fallback to period (end of sentence)
            split_pos = text.rfind('. ', start + chunk_size // 2, end)
            if split_pos != -1:
                split_pos += 1  # Include the period

        # If no clean boundary found, force cutoff at end
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunks.append(text[start:split_pos].strip())
        
        # Advance start position taking overlap into account
        start = max(split_pos - overlap, start + 1)

    return chunks
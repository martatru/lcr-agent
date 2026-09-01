import re
import logging

logger = logging.getLogger(__name__)

def prepare_full_text(raw_text: str) -> str:
    """
    Prepares the full text by cutting off the References section from the very end of the file,
    to avoid wasting tokens without cutting off the article content.
    """
    clean_text = re.sub(r'\r\n', '\n', raw_text)
    
    # Cut off the reference list from the end of the file (if present)
    ref_match = re.search(r'(?im)^\s*(?:#+\s*)?(?:\*\*)?\bReferences\b', clean_text)
    if ref_match:
        clean_text = clean_text[:ref_match.start()]
        
    return clean_text.strip()

# Alias for compatibility
extract_relevant_sections = prepare_full_text

def chunk_text(text: str, chunk_size: int = 60000, overlap: int = 5000) -> list[str]:
    """
    Splits text into overlapping windows for LLM processing.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
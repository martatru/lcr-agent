import pymupdf
import logging

logger = logging.getLogger(__name__)

def parse_pdf(pdf_path: str) -> str:
    """
    Direct text extraction using PyMuPDF - 100% bypasses Tesseract OCR.
    """
    try:
        doc = pymupdf.open(pdf_path)
        full_text = [page.get_text("text") for page in doc]
        return "\n\n".join(full_text)
    except Exception as error:
        logger.error(f"Local parsing error for {pdf_path}: {error}")
        return ""
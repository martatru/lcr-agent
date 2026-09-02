import re
import json
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse, quote

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_PDFS_DIR = Path("data/raw_pdfs")
RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def sanitize_filename(name: str) -> str:
    """Cleans paper title for safe use as a filename."""
    clean = re.sub(r'[\\/*?:"<>|]', '', name)
    clean = re.sub(r'\s+', '_', clean).strip()
    return clean[:80]

def resolve_doi_from_title_or_url(item: dict) -> str:
    """Extracts DOI directly or queries Crossref API using the title."""
    # 1. Check existing DOI or URL for DOI pattern
    url = item.get("url", "")
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', url)
    if doi_match:
        return doi_match.group(0)

    title = item.get("title", "").strip()
    if not title:
        return ""

    # 2. Query Crossref API to resolve exact DOI from paper title
    try:
        crossref_url = f"https://api.crossref.org/works?query.title={quote(title)}&rows=1"
        res = requests.get(crossref_url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            items = res.json().get("message", {}).get("items", [])
            if items and "DOI" in items[0]:
                found_doi = items[0]["DOI"]
                logger.info(f"Resolved DOI via Crossref: {found_doi}")
                return found_doi
    except Exception as err:
        logger.debug(f"Crossref lookup failed for '{title[:30]}': {err}")

    # 3. Fallback: Check PubMed HTML for citation_doi meta tag
    if "pubmed" in url:
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                match = re.search(r'<meta name="citation_doi" content="([^"]+)"', res.text)
                if match:
                    return match.group(1)
        except Exception:
            pass

    return ""

def get_publisher_pdf_url(doi: str) -> str:
    """Follows DOI redirect using Institutional IP and extracts publisher PDF URL."""
    if not doi:
        return ""

    doi_url = f"https://doi.org/{doi}"

    try:
        session = requests.Session()
        res = session.get(doi_url, headers=HEADERS, allow_redirects=True, timeout=15)
        
        if res.status_code == 200:
            # Check standard academic meta tags used by major publishers
            patterns = [
                r'<meta\s+(?:name|property)=["\']citation_pdf_url["\']\s+content=["\']([^"\'\s]+)["\']',
                r'<meta\s+content=["\']([^"\'\s]+)["\']\s+(?:name|property)=["\']citation_pdf_url["\']',
                r'<a\s+[^>]*href=["\']([^"\'\s]+\.pdf[^"\'\s]*)["\']'
            ]
            
            for pat in patterns:
                match = re.search(pat, res.text, re.IGNORECASE)
                if match:
                    pdf_url = match.group(1).replace("&amp;", "&")
                    if pdf_url.startswith("/"):
                        parsed = urlparse(res.url)
                        pdf_url = f"{parsed.scheme}://{parsed.netloc}{pdf_url}"
                    return pdf_url

            # Publisher-specific fallback rules (ScienceDirect / Elsevier / Nature / ACS)
            if "sciencedirect.com" in res.url or "elsevier.com" in res.url:
                pii_match = re.search(r'/article/pii/([A-Za-z0-9]+)', res.url)
                if pii_match:
                    return f"https://www.sciencedirect.com/science/article/pii/{pii_match.group(1)}/pdfft?isExport=true"

    except Exception as err:
        logger.debug(f"Publisher resolution failed for DOI {doi}: {err}")

    return ""

def download_pdf(pdf_url: str, output_path: Path) -> bool:
    """Downloads binary PDF file using institutional IP credentials."""
    try:
        session = requests.Session()
        res = session.get(pdf_url, headers=HEADERS, timeout=30, stream=True)
        res.raise_for_status()

        content_type = res.headers.get("Content-Type", "").lower()
        # Verify content is binary PDF
        if "pdf" in content_type or res.content.startswith(b"%PDF"):
            with open(output_path, "wb") as f:
                f.write(res.content)
            return True
    except Exception as err:
        logger.warning(f"Failed download from {pdf_url}: {err}")

    return False

def fetch_candidate_pdfs(candidates_file: str = "data/candidates.json"):
    """Downloads PDFs using Crossref DOI resolution and Institutional IP access."""
    path = Path(candidates_file)
    if not path.exists():
        logger.error(f"File {candidates_file} not found! Run search first.")
        return

    candidates = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Processing {len(candidates)} candidates via Crossref & Institutional IP...")

    downloaded_count = 0

    for idx, item in enumerate(candidates, start=1):
        title = item.get("title", "")
        filename = f"{sanitize_filename(title)}.pdf"
        out_pdf_path = RAW_PDFS_DIR / filename

        if out_pdf_path.exists():
            logger.info(f"[{idx}/{len(candidates)}] Already exists: {filename}")
            downloaded_count += 1
            continue

        logger.info(f"[{idx}/{len(candidates)}] Resolving DOI for: {title[:50]}...")

        # 1. Resolve DOI
        doi = resolve_doi_from_title_or_url(item)
        if not doi:
            logger.warning(f"Could not resolve DOI for: {title[:50]}")
            continue

        # 2. Extract Publisher PDF URL via Institutional IP
        pdf_url = get_publisher_pdf_url(doi)

        if pdf_url:
            logger.info(f"Found Publisher PDF link! Downloading...")
            if download_pdf(pdf_url, out_pdf_path):
                logger.info(f"✅ Successfully downloaded: {filename}")
                downloaded_count += 1
                continue

        logger.warning(f"⚠️ Publisher restricted or requires manual download for DOI: {doi}")

    logger.info(f"Done! {downloaded_count}/{len(candidates)} PDFs saved in {RAW_PDFS_DIR}")

if __name__ == "__main__":
    fetch_candidate_pdfs()
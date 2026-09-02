import re
import json
import logging
import requests
from pathlib import Path
from urllib.parse import quote

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_PDFS_DIR = Path("data/raw_pdfs")
RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def sanitize_filename(name: str) -> str:
    """Cleans paper title for safe use as a filename."""
    clean = re.sub(r'[\\/*?:"<>|]', '', name)
    clean = re.sub(r'\s+', '_', clean).strip()
    return clean[:80]


def resolve_doi_from_crossref(title: str) -> str:
    """Resolves paper DOI using Crossref REST API."""
    if not title or len(title.strip()) < 5:
        return ""
    try:
        url = f"https://api.crossref.org/works?query.title={quote(title)}&rows=1"
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            items = res.json().get("message", {}).get("items", [])
            if items and "DOI" in items[0]:
                return items[0]["DOI"]
    except Exception as err:
        logger.debug("Crossref lookup failed: %s", err)
    return ""


def get_pdf_from_unpaywall(doi: str) -> str:
    """Queries Unpaywall API for Open Access PDF mirrors hosted on institutional repositories."""
    if not doi:
        return ""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email=lcr_agent_biocurator@example.com"
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")
            if pdf_url:
                logger.info("Found Open Access PDF via Unpaywall: %s", pdf_url[:60])
                return pdf_url
    except Exception as err:
        logger.debug("Unpaywall API failed for DOI %s: %s", doi, err)
    return ""


def get_pdf_from_europe_pmc(doi: str, pmid: str) -> str:
    """Queries Europe PMC REST API for full-text PDF links."""
    query = f'DOI:"{doi}"' if doi else (f'EXT_ID:{pmid}' if pmid else "")
    if not query:
        return ""
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={quote(query)}&format=json&resultType=core"
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            results = res.json().get("resultList", {}).get("result", [])
            if results:
                url_list = results[0].get("fullTextUrlList", {}).get("fullTextUrl", [])
                for item in url_list:
                    if item.get("documentStyle") == "pdf":
                        pdf_url = item.get("url", "")
                        logger.info("Found PDF via Europe PMC: %s", pdf_url[:60])
                        return pdf_url
    except Exception as err:
        logger.debug("Europe PMC lookup failed: %s", err)
    return ""


def download_file_direct(url: str, output_path: Path) -> bool:
    """Downloads binary file via standard HTTP session."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        res.raise_for_status()
        if res.content.startswith(b"%PDF") or "pdf" in res.headers.get("Content-Type", "").lower():
            output_path.write_bytes(res.content)
            return True
    except Exception as err:
        logger.debug("Direct download failed from %s: %s", url, err)
    return False


def fetch_candidate_pdfs(candidates_file: str = "data/candidates.json"):
    """Multi-tiered pipeline for fetching candidate PDFs."""
    path = Path(candidates_file)
    if not path.exists():
        logger.error("File %s not found!", candidates_file)
        return

    candidates = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Processing %d candidates for PDF retrieval...", len(candidates))

    downloaded_count = 0
    missing_candidates = []

    for idx, item in enumerate(candidates, start=1):
        title = item.get("title", "").strip()
        pmid = item.get("pmid", "").strip()
        doi = item.get("doi", "").strip() or resolve_doi_from_crossref(title)

        filename = f"{sanitize_filename(title)}.pdf"
        out_pdf_path = RAW_PDFS_DIR / filename

        if out_pdf_path.exists():
            logger.info("[%d/%d] Already exists: %s", idx, len(candidates), filename)
            downloaded_count += 1
            continue

        logger.info("[%d/%d] Processing: %s...", idx, len(candidates), title[:50])

        pdf_url = get_pdf_from_unpaywall(doi) if doi else ""

        if not pdf_url:
            pdf_url = get_pdf_from_europe_pmc(doi, pmid)

        if pdf_url and download_file_direct(pdf_url, out_pdf_path):
            logger.info("Successfully downloaded: %s", filename)
            downloaded_count += 1
            continue

        logger.warning("Publisher paywall/Cloudflare blocked automatic fetch for: %s", title[:50])
        missing_candidates.append({
            "title": title,
            "doi": doi,
            "url": item.get("url", f"https://doi.org/{doi}" if doi else "")
        })

    logger.info("Done! %d/%d PDFs ready in %s", downloaded_count, len(candidates), RAW_PDFS_DIR)

    if missing_candidates:
        logger.info("Paywalled papers requiring manual drop into 'data/raw_pdfs/':")
        for m in missing_candidates:
            logger.info("  - %s | Link: %s", m["title"][:40], m["url"])


if __name__ == "__main__":
    fetch_candidate_pdfs()
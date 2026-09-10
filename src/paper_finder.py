"""Find, score, and download candidate papers about low-complexity
regions (LCRs) in proteins.

This module merges three previously separate scripts into a single
pipeline:

1. ``searxng_finder`` stage -- query SearXNG (PubMed engine) for
   candidate papers.
2. ``example_code`` scoring stage -- reuse the richer LCR/nucleic-acid
   term lists and disambiguation logic to score each abstract instead
   of relying on a single broad regex.
3. ``download_pdfs`` stage -- resolve a DOI for each surviving
   candidate and try to fetch an open-access PDF via Unpaywall and
   Europe PMC.

Run it directly to execute the full search -> score -> download
pipeline, or import the individual functions to use only part of it.
"""

import json
import logging
import re
from pathlib import Path
from urllib.parse import quote

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEARXNG_URL = "http://localhost:8080/search"  # change to your SearXNG URL

RAW_PDFS_DIR = Path("data/raw_pdfs")
CANDIDATES_FILE = Path("data/candidates.json")

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

UNPAYWALL_EMAIL = "lcr_agent_biocurator@example.com"

# Scoring weights used by ``score_abstract``.
SCORE_LCR_MATCH = 2
SCORE_BINDING = 1
SCORE_COORDINATES = 1
SCORE_AMBIGUOUS_RESOLVED_AS_PROTEIN = 1
PENALTY_HARD_MATCH = -3
PENALTY_NUCLEIC_MATCH = -3
CANDIDATE_SCORE_THRESHOLD = 2


# --------------------------------------------------------------------------
# LCR term lists (adapted from Sylwia)
# --------------------------------------------------------------------------

LCR_TERMS = [
    "polyglycine", "poly-glycine", "glycine-rich", "polyG", "poly-G",
    "G-rich", "poly(G)", "poly(glicyne)", "poly-Gly",
    "polyalanine", "poly-alanine", "alanine-rich", "polyA", "poly-A",
    "A-rich", "poly(A)", "poly(alanine)", "poly-Ala",
    "polyvaline", "poly-valine", "valine-rich", "polyV", "poly-V",
    "V-rich", "poly(V)", "poly(valine)", "poly-Val",
    "polyleucine", "poly-leucine", "leucine-rich", "polyL", "poly-L",
    "L-rich", "poly(L)", "poly(leucine)", "poly-Leu", "Leu-rich",
    "polyisoleucine", "poly-isoleucine", "isoleucine-rich", "polyI",
    "poly-I", "I-rich", "poly(I)", "poly(isoleucine)", "poly-Iso",
    "polyserine", "poly-serine", "serine-rich", "polyS", "poly-S",
    "S-rich", "poly(S)", "poly(serine)", "poly-Ser",
    "polythreonine", "poly-threonine", "threonine-rich", "polyT",
    "poly-T", "T-rich", "poly(T)", "poly(threonine)", "poly-Thr",
    "polycysteine", "poly-cysteine", "cysteine-rich", "polyC", "poly-C",
    "C-rich", "poly(C)", "poly(cysteine)", "poly-Cys", "Cys-rich",
    "polymethionine", "poly-methionine", "methionine-rich", "polyM",
    "poly-M", "M-rich", "poly(M)", "poly(methionine)", "poly-Met",
    "polyphenylalanine", "poly-phenylalanine", "phenylalanine-rich",
    "polyF", "poly-F", "F-rich", "poly(F)", "poly(phenylalanine)",
    "poly-Phe",
    "polytyrosine", "poly-tyrosine", "tyrosine-rich", "polyY", "poly-Y",
    "Y-rich", "poly(Y)", "poly(tyrosine)", "poly-Tyr",
    "polytryptophan", "poly-tryptophan", "tryptophan-rich", "polyW",
    "poly-W", "W-rich", "poly(W)", "poly(tryptophan)", "poly-Trp",
    "polyaspartic", "poly-polyaspartic", "polyaspartic-rich", "polyD",
    "poly-D", "D-rich", "poly(D)", "poly(aspartic)", "poly-Asp",
    "polyasparagine", "poly-asparagine", "asparagine-rich", "polyN",
    "poly-N", "N-rich", "poly(N)", "poly(asparagine)", "poly-Asn",
    "polyglutamic", "poly-glutamic", "glutamic-rich", "polyE", "poly-E",
    "E-rich", "poly(E)", "poly(glutamic)", "poly-Gln",
    "polyglutamine", "poly-glutamine", "glutamine-rich", "polyQ",
    "poly-Q", "Q-rich", "poly(Q)", "poly(glutamine)", "poly-Glu",
    "polylysine", "poly-lysine", "lysine-rich", "polyK", "poly-K",
    "K-rich", "poly(K)", "poly(lysine)", "poly-Lys",
    "polyarginine", "poly-arginine", "arginine-rich", "polyR", "poly-R",
    "R-rich", "poly(R)", "poly(arginine)", "poly-Arg",
    "polyhistidine", "poly-histidine", "histidine-rich", "polyH",
    "poly-H", "H-rich", "poly(H)", "poly(histidine)", "poly-His",
    "polyproline", "poly-proline", "proline-rich", "polyP", "poly-P",
    "P-rich", "poly(P)", "poly(proline)", "poly-Pro",
    "AT-rich", "RGG", "RG-rich", "GC-rich", "TG-rich", "CT-rich",
    "TC-rich", "poly-l-lysine", "poly-L-lysine", "serine/threonine-rich",
    "GT-rich", "poly-l-histidine", "poly-L-proline",
    "poly-L-glutamic acid", "poly-L-aspartic acid", "poly-L-arginine",
    "glycine/alanine", "proline/serine/threonine-rich",
    "cysteine/serine-rich", "proline/alanine-rich",
    "lysine/arginine-rich", "cysteine/histidine-rich",
    "arginine/lysine-rich", "Q/N-rich", "proline-serine-threonine-rich",
    "serine-threonine-rich", "low-complexity", "low complexity region",
    "low-complexity domain", "LCR", "LCD", "IDR", "IDP",
    "intrinsically disordered", "prion-like domain", "prion-like",
]

# Terms that look like LCR terms but belong to unrelated domains
# (materials science, gel electrophoresis, etc.) and should exclude a
# match rather than support it.
HARD_EXCLUSION_TERMS = [
    "Poly-L-lactide", "poly-l-lactide", "poly-N-", "polyacrylamide",
    "sulfate-polyacrylamide", "SDS-polyacrylamide", "acid-rich",
    "poly-ADP-ribose", "poly(amido amine)", "poly(amidoamine)",
    "poly-ADP-ribosylation", "poly-L-ornithine",
    "poly-N-acetyllactosamine", "polyI:C",
]

# Terms that unambiguously refer to nucleic acids, not proteins.
NUCLEIC_ACID_TERMS = [
    "poly(A)+", "poly(A)(+)", "poly-A+", "poly(A) tail", "poly(A) signal",
    "poly(A)-containing", "poly(A)-dependent", "poly(A)(+) RNA",
    "poly(A) containing RNA", "poly(A)-containing mRNA", "RNA-rich",
    "poly(A) nucleotides", "poly(U)", "polyadenylation", "polyadenylated",
    "AU-rich", "poly(A)-mRNA", "A + T-rich", "A:T-rich", "G + C-rich",
    "G+C-rich", "RNA poly(C)",
]

# Terms that could refer to either a protein or a nucleic acid
# depending on context (resolved via ``classify_ambiguous_term``).
AMBIGUOUS_TERMS = [
    "A-rich", "T-rich", "G-rich", "C-rich", "AT-rich", "GC-rich",
    "A/T-rich", "poly(A)", "CT-rich", "GT-rich",
]

# Nearby-word cues used to disambiguate ``AMBIGUOUS_TERMS``.
PROTEIN_CONTEXT_KEYWORDS = [
    "protein", "domain", "interaction domain", "amino acid", "residue",
    "protein region",
]
NUCLEIC_CONTEXT_KEYWORDS = [
    "dna", "rna", "mrna", "gene", "oligonucleotides", "promoter", "utr",
    "5'", "3'", "nucleotide",
]

PATTERN_BINDING = re.compile(
    r"\b(bind|binds|binding|bound|interact|interacts|interaction|"
    r"complex|associates|recognizes)\b",
    re.IGNORECASE,
)
PATTERN_COORDINATES = re.compile(
    r"(\b(residues?|aa|amino acids?|positions?)\s*\d+\s*[-\u2013\u2014\n"
    r"to]\s*\d+\b|\b\d+\s*[-\u2013\u2014]\s*\d+\s*(aa|residues)?\b)",
    re.IGNORECASE,
)


def _build_word_pattern(terms: list[str]) -> re.Pattern:
    """Build a case-insensitive, word-boundary regex from a term list."""
    escaped = [re.escape(term) for term in terms]
    return re.compile(
        r"(?<![\w()-])(" + "|".join(escaped) + r")(?![\w()-])", re.IGNORECASE
    )


PATTERN_LCR = _build_word_pattern(LCR_TERMS)
PATTERN_HARD = _build_word_pattern(HARD_EXCLUSION_TERMS)
PATTERN_NUCLEIC = _build_word_pattern(NUCLEIC_ACID_TERMS)
PATTERN_AMBIGUOUS = _build_word_pattern(AMBIGUOUS_TERMS)


# --------------------------------------------------------------------------
# Scoring / disambiguation (adapted from example_code.py)
# --------------------------------------------------------------------------

def _find_spans(pattern: re.Pattern, text: str) -> list[tuple[int, int]]:
    """Return the (start, end) character spans of every match."""
    return [(m.start(), m.end()) for m in pattern.finditer(text.lower())]


def _span_distance(
    spans_a: list[tuple[int, int]], spans_b: list[tuple[int, int]]
) -> float:
    """Return the smallest distance between two sets of text spans."""
    if not spans_a or not spans_b:
        return float("inf")
    midpoints_a = [(start + end) / 2 for start, end in spans_a]
    midpoints_b = [(start + end) / 2 for start, end in spans_b]
    return min(abs(a - b) for a in midpoints_a for b in midpoints_b)


def classify_ambiguous_term(term: str, text: str) -> str:
    """Classify an ambiguous term (e.g. ``A-rich``) as protein or
    nucleic-acid context, based on the nearest supporting keywords.
    """
    term_spans = _find_spans(re.compile(re.escape(term), re.IGNORECASE), text)

    protein_spans: list[tuple[int, int]] = []
    for keyword in PROTEIN_CONTEXT_KEYWORDS:
        protein_spans += _find_spans(
            re.compile(re.escape(keyword), re.IGNORECASE), text
        )

    nucleic_spans: list[tuple[int, int]] = []
    for keyword in NUCLEIC_CONTEXT_KEYWORDS:
        nucleic_spans += _find_spans(
            re.compile(re.escape(keyword), re.IGNORECASE), text
        )

    protein_distance = _span_distance(term_spans, protein_spans)
    nucleic_distance = _span_distance(term_spans, nucleic_spans)

    close_to_both = protein_distance < 25 and nucleic_distance < 25
    if close_to_both and protein_distance != nucleic_distance:
        return "PROTEIN" if protein_distance < nucleic_distance else "NUCLEIC"

    return "PROTEIN" if len(protein_spans) > len(nucleic_spans) else "NUCLEIC"


def score_abstract(title: str, abstract: str) -> dict:
    """Score a title/abstract pair for how likely it is to discuss an
    actual protein low-complexity region, rather than an unrelated
    "poly-X" material or a nucleic-acid feature.
    """
    full_text = f"{title} {abstract}"

    has_lcr = bool(PATTERN_LCR.search(full_text))
    has_hard = bool(PATTERN_HARD.search(full_text))
    has_nucleic = bool(PATTERN_NUCLEIC.search(full_text))
    has_binding = bool(PATTERN_BINDING.search(full_text))
    has_coordinates = bool(PATTERN_COORDINATES.search(full_text))

    ambiguous_terms = {m.lower() for m in PATTERN_AMBIGUOUS.findall(full_text)}
    resolved_as_protein = any(
        classify_ambiguous_term(term, full_text) == "PROTEIN"
        for term in ambiguous_terms
    )

    score = 0
    if has_lcr:
        score += SCORE_LCR_MATCH
    if has_binding:
        score += SCORE_BINDING
    if has_coordinates:
        score += SCORE_COORDINATES
    if resolved_as_protein:
        score += SCORE_AMBIGUOUS_RESOLVED_AS_PROTEIN
    if has_hard:
        score += PENALTY_HARD_MATCH
    if has_nucleic:
        score += PENALTY_NUCLEIC_MATCH

    is_candidate = has_lcr and score >= CANDIDATE_SCORE_THRESHOLD

    return {
        "is_candidate": is_candidate,
        "score": score,
        "has_lcr": has_lcr,
        "has_hard_exclusion": has_hard,
        "has_nucleic_exclusion": has_nucleic,
        "has_binding": has_binding,
        "has_coordinates": has_coordinates,
        "resolved_ambiguous_as_protein": resolved_as_protein,
    }


# --------------------------------------------------------------------------
# Stage 1: search SearXNG and build the candidate list
# --------------------------------------------------------------------------

def search_searxng(query: str, max_results: int = 50) -> list[dict]:
    """Fetch search results from a local SearXNG instance (JSON API)."""
    params = {
        "q": query,
        "format": "json",
        "engines": "pubmed",
        "pageno": 1,
    }
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("results", [])[:max_results]
    except requests.RequestException as error:
        logger.error("SearXNG connection error: %s", error)
        return []


def find_and_save_candidates(
    search_queries: list[str], output_file: Path = CANDIDATES_FILE
) -> list[dict]:
    """Search PubMed (via SearXNG), score every result, and save the
    candidates that pass ``CANDIDATE_SCORE_THRESHOLD``, sorted by
    score (best first).
    """
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for query in search_queries:
        logger.info("Searching SearXNG for query: '%s'", query)
        results = search_searxng(query)

        for item in results:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = item.get("title", "")
            abstract = item.get("content", "")
            evaluation = score_abstract(title, abstract)

            if evaluation["is_candidate"]:
                logger.info(
                    "Match (score=%d) [lcr=%s binding=%s coords=%s] -> %s",
                    evaluation["score"],
                    evaluation["has_lcr"],
                    evaluation["has_binding"],
                    evaluation["has_coordinates"],
                    title[:60],
                )
                all_results.append({
                    "title": title,
                    "abstract": abstract,
                    "url": url,
                    "engine": item.get("engine", "pubmed"),
                    "score_flags": evaluation,
                })

    all_results.sort(key=lambda r: r["score_flags"]["score"], reverse=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Found %d candidate papers. Saved to: %s", len(all_results), output_file
    )
    return all_results


# --------------------------------------------------------------------------
# Stage 2: resolve DOIs and download PDFs for the saved candidates
# --------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Clean a paper title for safe use as a filename."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r"\s+", "_", clean).strip()
    return clean[:80]


def resolve_doi_from_crossref(title: str) -> str:
    """Resolve a paper's DOI using the Crossref REST API."""
    if not title or len(title.strip()) < 5:
        return ""
    try:
        url = f"https://api.crossref.org/works?query.title={quote(title)}&rows=1"
        response = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
            if items and "DOI" in items[0]:
                return items[0]["DOI"]
    except requests.RequestException as error:
        logger.debug("Crossref lookup failed: %s", error)
    return ""


def get_pdf_from_unpaywall(doi: str) -> str:
    """Query Unpaywall for an open-access PDF mirror for ``doi``."""
    if not doi:
        return ""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
        response = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")
            if pdf_url:
                logger.info("Found Open Access PDF via Unpaywall: %s", pdf_url[:60])
                return pdf_url
    except requests.RequestException as error:
        logger.debug("Unpaywall API failed for DOI %s: %s", doi, error)
    return ""


def get_pdf_from_europe_pmc(doi: str, pmid: str) -> str:
    """Query Europe PMC for a full-text PDF link for ``doi``/``pmid``."""
    query = f'DOI:"{doi}"' if doi else (f"EXT_ID:{pmid}" if pmid else "")
    if not query:
        return ""
    try:
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={quote(query)}&format=json&resultType=core"
        )
        response = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if response.status_code == 200:
            results = response.json().get("resultList", {}).get("result", [])
            if results:
                url_list = results[0].get("fullTextUrlList", {}).get(
                    "fullTextUrl", []
                )
                for item in url_list:
                    if item.get("documentStyle") == "pdf":
                        pdf_url = item.get("url", "")
                        logger.info("Found PDF via Europe PMC: %s", pdf_url[:60])
                        return pdf_url
    except requests.RequestException as error:
        logger.debug("Europe PMC lookup failed: %s", error)
    return ""


def download_file_direct(url: str, output_path: Path) -> bool:
    """Download a binary file over plain HTTP and verify it is a PDF."""
    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=30, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if response.content.startswith(b"%PDF") or "pdf" in content_type:
            output_path.write_bytes(response.content)
            return True
    except requests.RequestException as error:
        logger.debug("Direct download failed from %s: %s", url, error)
    return False


def fetch_candidate_pdfs(candidates_file: Path = CANDIDATES_FILE) -> None:
    """Download a PDF for every saved candidate, highest score first."""
    if not candidates_file.exists():
        logger.error("File %s not found!", candidates_file)
        return

    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = json.loads(candidates_file.read_text(encoding="utf-8"))
    candidates.sort(
        key=lambda c: c.get("score_flags", {}).get("score", 0), reverse=True
    )
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

        logger.warning(
            "Publisher paywall/Cloudflare blocked automatic fetch for: %s",
            title[:50],
        )
        missing_candidates.append({
            "title": title,
            "doi": doi,
            "url": item.get("url", f"https://doi.org/{doi}" if doi else ""),
        })

    logger.info(
        "Done! %d/%d PDFs ready in %s",
        downloaded_count, len(candidates), RAW_PDFS_DIR,
    )

    if missing_candidates:
        logger.info("Paywalled papers requiring manual drop into '%s':", RAW_PDFS_DIR)
        for missing in missing_candidates:
            logger.info("  - %s | Link: %s", missing["title"][:40], missing["url"])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_pipeline(search_queries: list[str]) -> None:
    """Run the full search -> score -> download pipeline."""
    find_and_save_candidates(search_queries)
    fetch_candidate_pdfs()


if __name__ == "__main__":
    default_queries = [
        '("low complexity region" OR "low complexity domain" OR '
        '"intrinsically disordered") AND (binding OR interaction) AND ' 
        '(residues OR "amino acids")',
        '("prion-like domain" OR "LCD" OR "IDR") AND (RNA-binding OR '
        'DNA-binding OR protein-binding)',
    ]
    run_pipeline(default_queries)
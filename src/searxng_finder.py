import re
import json
import logging
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# SearXNG Configuration
SEARXNG_URL = "http://localhost:8080/search"  # Change to your SearXNG URL

# RegEx patterns for rule-based filtering
PATTERN_LCR = re.compile(
    r'\b(LCRs?|LCDs?|IDRs?|IDPs?|PLDs?|low-complexity|low complexity|disordered|prion-like|intrinsically disordered|repeat|repeats)\b',
    re.IGNORECASE
)

PATTERN_BINDING = re.compile(
    r'\b(bind|binds|binding|bound|interact|interacts|interaction|complex|associates|recognizes)\b',
    re.IGNORECASE
)

PATTERN_COORDINATES = re.compile(
    r'(\b(residues?|aa|amino acids?|positions?)\s*\d+\s*[-–—\nto]\s*\d+\b|\b\d+\s*[-–—]\s*\d+\s*(aa|residues)?\b)',
    re.IGNORECASE
)


def search_searxng(query: str, max_results: int = 50) -> list[dict]:
    """Fetches search results directly from SearXNG in JSON format."""
    params = {
        "q": query,
        "format": "json",
        "engines": "pubmed",
        "pageno": 1
    }
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])[:max_results]
    except Exception as e:
        logger.error(f"SearXNG connection error: {e}")
        return []


def is_promising_abstract(title: str, abstract: str) -> dict:
    """Checks without AI whether the article meets LCR + Binding + Coordinates criteria."""
    full_text = f"{title} {abstract}"

    has_lcr = bool(PATTERN_LCR.search(full_text))
    has_binding = bool(PATTERN_BINDING.search(full_text))
    has_coords = bool(PATTERN_COORDINATES.search(full_text))

    # An article is a candidate if it contains LCR terms, binding activity, and explicit coordinates
    is_candidate = has_lcr and has_binding and has_coords

    return {
        "is_candidate": is_candidate,
        "has_lcr": has_lcr,
        "has_binding": has_binding,
        "has_coords": has_coords
    }


def find_and_save_candidates(search_queries: list[str], output_file: str = "data/candidates.json"):
    """Searches PubMed via SearXNG and saves pre-filtered candidate papers."""
    all_results = []
    seen_urls = set()

    for query in search_queries:
        logger.info(f"Searching SearXNG for query: '{query}'")
        results = search_searxng(query)

        for item in results:
            url = item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = item.get("title", "")
            abstract = item.get("content", "")

            eval_res = is_promising_abstract(title, abstract)

            if eval_res["is_candidate"]:
                logger.info(f"0-Token Match! [LCR={eval_res['has_lcr']}, Bind={eval_res['has_binding']}, Coords={eval_res['has_coords']}] -> {title[:60]}...")
                all_results.append({
                    "title": title,
                    "abstract": abstract,
                    "url": url,
                    "engine": item.get("engine", "pubmed"),
                    "score_flags": eval_res
                })

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    logger.info(f"Found {len(all_results)} candidate papers. Saved to: {output_file}")
    return all_results


if __name__ == "__main__":
    # Constructing targeted PubMed search queries for SearXNG
    queries = [
        '("low complexity region" OR "low complexity domain" OR "intrinsically disordered") AND (binding OR interaction) AND (residues OR "amino acids")',
        '("prion-like domain" OR "LCD" OR "IDR") AND (RNA-binding OR DNA-binding OR protein-binding) AND "1.."'
    ]

    find_and_save_candidates(queries)
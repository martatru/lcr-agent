"""Client for interacting with the self-hosted PlaToLoCo LCR detection API."""

import logging
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

PLATOLOCO_API_URL = "http://127.0.0.1:5002/restapi"

# Standardized mapping for all 8 PlaToLoCo predictor methods
METHOD_LABELS: Dict[str, str] = {
    # SEG variants
    "SEG": "SEG",
    "seg": "SEG",
    "seg_default": "SEG",
    "SEG_default": "SEG",
    "SEG_intermediate": "SEG_intermediate",
    "seg_intermediate": "SEG_intermediate",
    "SEG_strict": "SEG_strict",
    "seg_strict": "SEG_strict",
    # CAST
    "CAST": "CAST",
    "cast": "CAST",
    # fLPS variants
    "fLPS": "fLPS",
    "FLPS": "fLPS",
    "flps": "fLPS",
    "fLPS_strict": "fLPS_strict",
    "flps_strict": "fLPS_strict",
    # SIMPLE & GBSC
    "SIMPLE": "SIMPLE",
    "simple": "SIMPLE",
    "GBSC": "GBSC",
    "gbsc": "GBSC",
}


class PlatoLoCoClient:
    """Handles submission, polling, and parsing of all 8 LCR methods from PlaToLoCo."""

    def __init__(
        self,
        api_url: str = PLATOLOCO_API_URL,
        poll_interval: int = 2,
        timeout: int = 30,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout

    def is_service_available(self) -> bool:
        """Check if the local PlaToLoCo REST API endpoint is responsive."""
        try:
            response = requests.get(f"{self.api_url}/", timeout=3)
            return response.status_code < 500
        except requests.RequestException:
            return False

    def fetch_intervals(self, sequence: str, protein_id: str) -> List[Dict[str, Any]]:
        """Submit protein sequence and retrieve detected LCR intervals across all methods."""
        if not self.is_service_available():
            logger.warning(
                "PlaToLoCo service at %s is offline. Skipping interval detection.",
                self.api_url,
            )
            return []

        payload = self._make_payload(sequence, protein_id)

        try:
            response = requests.put(
                f"{self.api_url}/query", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            token = response.json().get("token")

            if not token:
                logger.error("PlaToLoCo failed to return a valid job token.")
                return []

            # Poll until task completion
            while True:
                status_res = requests.get(
                    f"{self.api_url}/job/{token}", timeout=self.timeout
                )
                status_res.raise_for_status()
                status = status_res.json().get("status")

                if status == "FINISHED":
                    break
                if status == "ERROR":
                    logger.error(
                        "PlaToLoCo processing error for protein: %s", protein_id
                    )
                    return []

                time.sleep(self.poll_interval)

            protein_res = requests.get(
                f"{self.api_url}/proteins/{token}", timeout=self.timeout
            )
            protein_res.raise_for_status()
            proteins = protein_res.json().get("proteins", [])

            if not proteins:
                return []

            p_internal_id = proteins[0]["id"]
            details_res = requests.get(
                f"{self.api_url}/proteins/{token}/{p_internal_id}",
                timeout=self.timeout,
            )
            details_res.raise_for_status()
            details = details_res.json()

            protein_seq = details.get("sequence", "")
            intervals: List[Dict[str, Any]] = []

            for result in details.get("data", {}).get("wrapper", []):
                raw_method = result.get("method", "")
                regions = result.get("regions", [])
                output_method = METHOD_LABELS.get(raw_method, raw_method)

                for region in regions:
                    start = int(region["beg"])
                    end = int(region["end"])
                    sub_seq = protein_seq[start - 1 : end] if protein_seq else ""

                    intervals.append({
                        "method": output_method,
                        "start": start,
                        "end": end,
                        "length": end - start + 1,
                        "sequence": sub_seq,
                        "description": region.get("description", ""),
                    })

            return intervals

        except requests.RequestException as error:
            logger.warning(
                "PlaToLoCo request failed for %s: %s. Continuing without track data.",
                protein_id,
                error,
            )
            return []

    def _make_payload(self, sequence: str, protein_id: str) -> Dict[str, Any]:
        """Construct JSON query payload enabling all 8 prediction algorithms."""
        formatted_fasta = f">{protein_id}\n{sequence.strip()}\n"
        flps_params = {
            "min_tract_len": 15,
            "max_tract_len": 500,
            "pval": 0.001,
            "regions": {"single": True, "multiple": True, "whole": False},
        }

        return {
            "name": protein_id,
            "sequences": formatted_fasta,
            "methods": {
                "seg_default": True,
                "seg_intermediate": True,
                "seg_strict": True,
                "cast": True,
                "flps": True,
                "flps_strict": True,
                "simple": True,
                "gbsc": True,
            },
            "enrichment": {
                "pfam": False,
                "phobius": False,
                "aafrequency": False,
            },
            "params": {
                "seg_default": {},
                "seg_strict": {},
                "seg_intermediate": {"window": 15, "k1": 1.9, "k2": 2.5},
                "cast": {"threshold": 40, "matrix": 1},
                "flps": flps_params,
                "flps_strict": flps_params,
                "simple": {},
                "gbsc": {},
            },
        }
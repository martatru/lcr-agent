"""Client for interacting with the self-hosted PlaToLoCo LCR detection API."""

import logging
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

PLATOLOCO_API_URL = "http://127.0.0.1:5002/restapi"

METHOD_LABELS: Dict[str, str] = {
    "SEG": "SEG",
    "seg": "SEG",
    "seg_default": "SEG",
    "SEG_default": "SEG",
    "SEG_intermediate": "SEG_intermediate",
    "seg_intermediate": "SEG_intermediate",
    "SEG_strict": "SEG_strict",
    "seg_strict": "SEG_strict",
    "CAST": "CAST",
    "cast": "CAST",
    "fLPS": "fLPS",
    "FLPS": "fLPS",
    "flps": "fLPS",
    "fLPS_strict": "fLPS_strict",
    "flps_strict": "fLPS_strict",
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

            protein_summary = proteins[0]
            p_internal_id = protein_summary.get("id")
            intervals: List[Dict[str, Any]] = []
            seen_intervals = set()

            if p_internal_id is not None:
                try:
                    details_res = requests.get(
                        f"{self.api_url}/proteins/{token}/{p_internal_id}",
                        timeout=self.timeout,
                    )
                    if details_res.status_code == 200:
                        details = details_res.json()
                        protein_seq = details.get("sequence", "")

                        for result in details.get("data", {}).get("wrapper", []):
                            raw_method = result.get("method", "")
                            output_method = METHOD_LABELS.get(
                                raw_method,
                                METHOD_LABELS.get(raw_method.lower(), raw_method),
                            )

                            for region in result.get("regions", []):
                                try:
                                    start = int(region["beg"])
                                    end = int(region["end"])
                                    sub_seq = (
                                        protein_seq[start - 1 : end]
                                        if protein_seq
                                        else ""
                                    )
                                    pair_key = (output_method, start, end)

                                    if pair_key not in seen_intervals:
                                        seen_intervals.add(pair_key)
                                        intervals.append({
                                            "method": output_method,
                                            "start": start,
                                            "end": end,
                                            "length": end - start + 1,
                                            "sequence": sub_seq,
                                            "description": region.get(
                                                "description", ""
                                            ),
                                        })
                                except (KeyError, ValueError, TypeError):
                                    pass
                except requests.RequestException as err:
                    logger.warning("Detail parsing request error: %s", err)

            for key, val in protein_summary.items():
                std_method = METHOD_LABELS.get(key) or METHOD_LABELS.get(key.lower())
                if std_method and isinstance(val, list) and val:
                    for reg in val:
                        if isinstance(reg, list) and len(reg) == 2:
                            try:
                                start = int(reg[0])
                                end = int(reg[1])
                                pair_key = (std_method, start, end)

                                if pair_key not in seen_intervals:
                                    seen_intervals.add(pair_key)
                                    intervals.append({
                                        "method": std_method,
                                        "start": start,
                                        "end": end,
                                        "length": end - start + 1,
                                        "sequence": sequence[start - 1 : end],
                                        "description": "",
                                    })
                            except (ValueError, TypeError):
                                pass

            return sorted(intervals, key=lambda x: (x["method"], x["start"]))

        except requests.RequestException as error:
            logger.warning(
                "PlaToLoCo request failed for %s: %s. Continuing without track data.",
                protein_id,
                error,
            )
            return []

    def _make_payload(self, sequence: str, protein_id: str) -> Dict[str, Any]:
        """Construct JSON query payload with explicit parameters for all 8 algorithms."""
        formatted_fasta = f">{protein_id}\n{sequence.strip()}\n"

        seg_default_params = {
            "window": 12,
            "locut": 2.2,
            "hicut": 2.5,
            "k1": 2.2,
            "k2": 2.5,
        }

        flps_params = {
            "min_tract_len": 15,
            "max_tract_len": 500,
            "pval": 0.001,
            "regions": {"single": True, "multiple": True, "whole": False},
        }

        simple_params = {
            "score_mono": 1.0,
            "score_di": 1.0,
            "score_tri": 1.0,
            "score_tetra": 1.0,
            "score_penta": 1.0,
            "score_hexa": 1.0,
            "score_hepta": 1.0,
            "score_octa": 1.0,
            "score_nona": 1.0,
            "score_deca": 1.0,
            "window": 20,
            "num_of_rand": 1000,
            "rand_method": 1,
            "stringency": 1.0,
        }

        gbsc_params = {
            "score": 0,
            "distance": 0,
        }

        payload_params = {
            "seg": seg_default_params,
            "seg_default": seg_default_params,
            "seg_strict": {},
            "seg_intermediate": {"window": 15, "k1": 1.9, "k2": 2.5},
            "cast": {"threshold": 40, "matrix": 1},
            "flps": flps_params,
            "flps_strict": flps_params,
            "simple": simple_params,
            "gbsc": gbsc_params,
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
            "params": payload_params,
        }
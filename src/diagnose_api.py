"""PlaToLoCo API Diagnostic Suite for all 8 prediction methods."""

import json
import time
import uuid
import requests

PLATOLOCO_API_URL = "http://127.0.0.1:5002/restapi"
TEST_UNIPROT_ID = "G2TRN4"

ALL_METHODS = [
    "seg_default",
    "seg_intermediate",
    "seg_strict",
    "cast",
    "flps",
    "flps_strict",
    "simple",
    "gbsc",
]


def fetch_sequence(uniprot_id: str) -> str:
    """Fetch protein sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json().get("sequence", {}).get("value", "")
    except Exception as err:
        print(f"Error fetching sequence from UniProt: {err}")
    return ""


def run_single_method_test(sequence: str, method_name: str) -> None:
    """Submit a single method query to PlaToLoCo API and inspect status."""
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
    }

    gbsc_params = {
        "score": 0,
        "distance": 0,
    }

    methods_flag = {m: (m == method_name) for m in ALL_METHODS}

    payload = {
        "name": f"diag_{method_name}_{uuid.uuid4().hex[:4]}",
        "sequences": f">{TEST_UNIPROT_ID}\n{sequence}\n",
        "methods": methods_flag,
        "enrichment": {"pfam": False, "phobius": False, "aafrequency": False},
        "params": {
            "seg": seg_default_params,
            "seg_default": seg_default_params,
            "seg_strict": {},
            "seg_intermediate": {"window": 15, "k1": 1.9, "k2": 2.5},
            "cast": {"threshold": 40, "matrix": 1},
            "flps": flps_params,
            "flps_strict": flps_params,
            "simple": simple_params,
            "gbsc": gbsc_params,
        },
    }

    print(f"\n[+] Testing method: {method_name}")
    try:
        res = requests.put(f"{PLATOLOCO_API_URL}/query", json=payload, timeout=5)
        if res.status_code != 200:
            print(f"    [-] Submission FAILED (HTTP {res.status_code}): {res.text}")
            return

        token = res.json().get("token")
        print(f"    Job token: {token}")

        status = "UNKNOWN"
        for _ in range(15):
            st_res = requests.get(f"{PLATOLOCO_API_URL}/job/{token}", timeout=5)
            if st_res.status_code == 200:
                status = st_res.json().get("status")
                if status in ("FINISHED", "ERROR"):
                    break
            time.sleep(1)

        print(f"    Status: {status}")

        if status == "FINISHED":
            list_res = requests.get(
                f"{PLATOLOCO_API_URL}/proteins/{token}", timeout=5
            )
            if list_res.status_code == 200:
                proteins = list_res.json().get("proteins", [])
                if proteins:
                    p_id = proteins[0].get("id")
                    details_res = requests.get(
                        f"{PLATOLOCO_API_URL}/proteins/{token}/{p_id}",
                        timeout=5,
                    )
                    if details_res.status_code == 200:
                        wrapper = (
                            details_res.json()
                            .get("data", {})
                            .get("wrapper", [])
                        )
                        print(f"    Result: {len(wrapper)} regions found in wrapper.")
        elif status == "ERROR":
            print(f"    [-] Execution ERROR in PlaToLoCo for '{method_name}'")

    except Exception as err:
        print(f"    [-] Exception: {err}")


def main() -> None:
    """Run diagnostic suite across all 8 methods."""
    print(f"Fetching sequence for {TEST_UNIPROT_ID}...")
    sequence = fetch_sequence(TEST_UNIPROT_ID)
    if not sequence:
        print("Failed to retrieve sequence. Aborting diagnostic.")
        return

    print(
        f"Sequence fetched ({len(sequence)} aa). Testing API endpoint: {PLATOLOCO_API_URL}"
    )
    for method in ALL_METHODS:
        run_single_method_test(sequence, method)


if __name__ == "__main__":
    main()
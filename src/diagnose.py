"""Diagnostic script to test individual PlaToLoCo methods against the local API."""

import time
import uuid
import requests

API_URL = "http://127.0.0.1:5002/restapi"

METHODS_TO_TEST = [
    "seg_default",
    "seg_intermediate",
    "seg_strict",
    "cast",
    "flps",
    "flps_strict",
    "simple",
    "gbsc",
]


def run_diagnostic() -> None:
    """Test each PlaToLoCo method independently to identify backend failures."""
    test_sequence = ">diagnostic_seq\nMSQSEDVSSKLEKLLKKLGVEDKEDKEE\n"
    print("Starting PlaToLoCo individual method diagnostics...\n")

    for method in METHODS_TO_TEST:
        payload = {
            "name": f"diag_{method}_{uuid.uuid4().hex[:4]}",
            "sequences": test_sequence,
            "methods": {m: (m == method) for m in METHODS_TO_TEST},
            "enrichment": {"pfam": False, "phobius": False, "aafrequency": False},
            "params": {
                "seg_default": {},
                "seg_strict": {},
                "seg_intermediate": {"window": 15, "k1": 1.9, "k2": 2.5},
                "cast": {"threshold": 40, "matrix": 1},
                "flps": {
                    "min_tract_len": 5,
                    "max_tract_len": 500,
                    "pval": 0.001,
                    "regions": {"single": True, "multiple": True, "whole": False},
                },
                "flps_strict": {
                    "min_tract_len": 5,
                    "max_tract_len": 500,
                    "pval": 0.001,
                    "regions": {"single": True, "multiple": True, "whole": False},
                },
                "simple": {},
                "gbsc": {},
            },
        }

        try:
            res = requests.put(f"{API_URL}/query", json=payload, timeout=5)
            if res.status_code != 200:
                print(f"[-] Method '{method}': HTTP {res.status_code} on submission")
                continue

            token = res.json().get("token")
            if not token:
                print(f"[-] Method '{method}': No token returned")
                continue

            status = "PROCESSING"
            for _ in range(15):
                st_res = requests.get(f"{API_URL}/job/{token}", timeout=5)
                if st_res.status_code == 200:
                    status = st_res.json().get("status")
                    if status in ("FINISHED", "ERROR"):
                        break
                time.sleep(1)

            if status == "FINISHED":
                prot_res = requests.get(f"{API_URL}/proteins/{token}", timeout=5)
                if prot_res.status_code == 200:
                    print(f"[+] Method '{method}': SUCCESS")
                else:
                    print(f"[-] Method '{method}': Finished but failed to fetch proteins")
            else:
                print(f"[-] Method '{method}': Failed with status {status} (Check Docker logs)")

        except Exception as err:
            print(f"[-] Method '{method}': Exception occurred -> {err}")


if __name__ == "__main__":
    run_diagnostic()
import json
from pathlib import Path

def process_results(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    structured_results = []

    for entry in lines:
        filename = entry["file"]
        annotations = entry["annotations"]
        
        verified = []
        suspicious = []
        seen_verified = set()
        seen_suspicious = set()

        for ann in annotations:
            p_name = ann.get("protein_name", "").strip()
            start = str(ann.get("start_of_annotation", "")).strip()
            end = str(ann.get("end_of_annotation", "")).strip()
            
            has_numbers = start.isdigit() and end.isdigit()

            if has_numbers:
                if p_name not in seen_verified:
                    seen_verified.add(p_name)
                    verified.append({
                        "protein_name": p_name,
                        "organism": ann.get("organism", "Unspecified"),
                        "start_of_annotation": start,
                        "end_of_annotation": end,
                        "proposed_function": ann.get("proposed_function", ""),
                        "evidence": ann.get("evidence", "")
                    })
            else:
                if p_name not in seen_suspicious and p_name not in seen_verified:
                    seen_suspicious.add(p_name)
                    suspicious.append({
                        "protein_name": p_name,
                        "organism": ann.get("organism", "Unspecified"),
                        "evidence": ann.get("evidence", ""),
                        "flag": "Qualitative mention only (no exact numbers in text)"
                    })

        structured_results.append({
            "file": filename,
            "verified_lcrs": verified,
            "suspicious_mentions": suspicious
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for item in structured_results:
            f.write(json.dumps(item, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    process_results("data/processed/final_results.jsonl", "data/processed/final_results_clean.json")
    print("Zapisano uporządkowane wyniki do: data/processed/final_results_clean.json")
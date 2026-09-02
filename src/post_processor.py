import json
import csv
from pathlib import Path


def process_results(
    input_file: str = "data/processed/final_results.jsonl",
    verified_output_file: str = "data/processed/verified_lcrs.json",
    qualitative_output_file: str = "data/processed/qualitative_mentions.json",
    qualitative_csv_file: str = "data/processed/qualitative_mentions.csv"
):
    """Processes JSONL output into clean JSON and CSV files separated by coordinate availability."""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} does not exist!")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    all_verified = []
    all_qualitative = []

    for entry in lines:
        filename = entry.get("file", "Unspecified")
        annotations = entry.get("annotations", [])

        seen_verified = set()
        seen_qualitative = set()

        for ann in annotations:
            p_name = ann.get("protein_name", "").strip()
            start = str(ann.get("start_of_annotation", "")).strip()
            end = str(ann.get("end_of_annotation", "")).strip()

            has_numbers = start.isdigit() and end.isdigit()

            if has_numbers:
                if p_name not in seen_verified:
                    seen_verified.add(p_name)
                    all_verified.append({
                        "file": filename,
                        "protein_name": p_name,
                        "organism": ann.get("organism", "Unspecified"),
                        "start_of_annotation": int(start),
                        "end_of_annotation": int(end),
                        "binding_target": ann.get("binding_target", "Unspecified"),
                        "proposed_function": ann.get("proposed_function", ""),
                        "evidence": ann.get("evidence", ""),
                        "curator_note": ann.get("curator_note", "")
                    })
            else:
                if p_name not in seen_qualitative and p_name not in seen_verified:
                    seen_qualitative.add(p_name)
                    all_qualitative.append({
                        "file": filename,
                        "protein_name": p_name,
                        "organism": ann.get("organism", "Unspecified"),
                        "binding_target": ann.get("binding_target", "Unspecified"),
                        "evidence": ann.get("evidence", ""),
                        "curator_note": ann.get("curator_note", ""),
                        "flag": "Qualitative mention only (no exact numbers in text)"
                    })

    v_path = Path(verified_output_file)
    v_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(all_verified, f, ensure_ascii=False, indent=2)

    q_path = Path(qualitative_output_file)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(all_qualitative, f, ensure_ascii=False, indent=2)

    if qualitative_csv_file and all_qualitative:
        c_path = Path(qualitative_csv_file)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file", "protein_name", "organism", "binding_target",
            "evidence", "curator_note", "flag"
        ]
        with open(c_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_qualitative)

    print(f"Saved {len(all_verified)} verified LCRs to: {verified_output_file}")
    print(f"Saved {len(all_qualitative)} qualitative mentions to: {qualitative_output_file}")


if __name__ == "__main__":
    process_results()
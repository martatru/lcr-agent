"""
Benchmark Dataset Builder for LCR-Agent.

1. Reads manual annotations from data/previously_annotated_lcr/list_of_annotations.csv
2. Generates a unique list of DOIs (data/unique_dois.txt) for manual paper downloading.
3. Builds a structured ground truth JSON dataset (data/ground_truth_benchmark.json)
   formatted for accuracy metrics (F1-score & Mean IoU).
"""

import json
from pathlib import Path
import pandas as pd


def build_benchmark_files(
    csv_path: str = "data/previously_annotated_lcr/list_of_annotations.csv",
    dois_output_path: str = "data/unique_dois.txt",
    benchmark_output_path: str = "data/ground_truth_benchmark.json",
):
    input_file = Path(csv_path)
    if not input_file.exists():
        print(f"Błąd: Plik {csv_path} nie istnieje!")
        return

    # Wczytanie danych z obsługą możliwych separatorów (przecinek / tabulacja)
    try:
        df = pd.read_csv(input_file)
        if len(df.columns) == 1:
            df = pd.read_csv(input_file, sep="\t")
    except Exception as err:
        print(f"Błąd podczas wczytywania CSV: {err}")
        return

    # 1. Ekstrakcja unikalnych DOI / Source ID
    source_col = "Source ID" if "Source ID" in df.columns else df.columns[9]
    unique_dois = (
        df[source_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    dois_file = Path(dois_output_path)
    dois_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dois_file, "w", encoding="utf-8") as f:
        for doi in unique_dois:
            f.write(f"{doi}\n")

    print(f"Zapisano {len(unique_dois)} unikalnych DOI w: {dois_output_path}")

    # 2. Budowa struktury JSON do walidacji
    benchmark_entries = []
    for idx, row in df.iterrows():
        # Pobieranie granic adnotacji eksperymentalnej
        start_annot = str(row.get("Start of annotation", "")).strip()
        end_annot = str(row.get("End of annotation", "")).strip()

        # Konwersja na liczby całkowite jeśli to możliwe
        start_valid = start_annot.isdigit()
        end_valid = end_annot.isdigit()

        # Określenie statusu adnotacji: Verified vs Requires Manual Check
        if start_valid and end_valid:
            status = "Verified"
            start_val = int(start_annot)
            end_val = int(end_annot)
        else:
            status = "Requires Manual Check"
            start_val = "Unspecified"
            end_val = "Unspecified"

        # Mapowanie typu LCR / celu wiązania
        binding_target = str(
            row.get("AnnotationCategory", row.get("LCR type", "Unspecified"))
        ).strip()

        entry = {
            "annotation_id": idx + 1,
            "source_id": str(row.get("Source ID", "N/A")).strip(),
            "uniprot_id": str(row.get("UniprotID", "N/A")).strip(),
            "gene_name": str(row.get("Gene name", "N/A")).strip(),
            "protein_name": str(row.get("Name", "N/A")).strip(),
            "protein_length": (
                int(row.get("Protein length"))
                if pd.notna(row.get("Protein length")) and str(row.get("Protein length")).isdigit()
                else "Unspecified"
            ),
            "organism": str(row.get("Organism", "N/A")).strip(),
            "start_of_annotation": start_val,
            "end_of_annotation": end_val,
            "binding_target": binding_target if binding_target != "nan" else "Unspecified",
            "curation_status": status,
            "evidence": str(row.get("Source", "")).strip(),
            "gene_ontology": str(row.get("Gene Ontology of category", "N/A")).strip(),
        }
        benchmark_entries.append(entry)

    json_file = Path(benchmark_output_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_entries, f, ensure_ascii=False, indent=2)

    print(f"Przekształcono {len(benchmark_entries)} adnotacji w: {benchmark_output_path}")


if __name__ == "__main__":
    build_benchmark_files()
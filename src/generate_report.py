import json
import urllib.parse
import requests
from pathlib import Path


def fetch_uniprot_metadata(protein_name: str, organism: str) -> dict:
    """Fetches protein metadata from the UniProt REST API during report generation."""
    query = f"({protein_name}) AND (organism_name:{organism})"
    url = f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(query)}&format=json&size=1"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.json().get("results"):
            res = response.json()["results"][0]
            
            go_terms = [
                ref["id"] for ref in res.get("uniProtKBCrossReferences", [])
                if ref["database"] == "GO"
            ]
            
            genes = res.get("genes", [{}])
            gene_name = genes[0].get("geneName", {}).get("value", "N/A") if genes else "N/A"
            full_name = res.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", protein_name)
            
            return {
                "uniprot_id": res["primaryAccession"],
                "gene_name": gene_name,
                "full_name": full_name,
                "length": res.get("sequence", {}).get("length", "N/A"),
                "go_terms": go_terms
            }
    except Exception as error:
        print(f"UniProt query error for {protein_name}: {error}")
    
    return {}


def generate_html_report(
    input_file: str = "data/processed/verified_lcrs.json",
    output_html: str = "data/processed/database_report.html"
):
    """Fetches UniProt data on-the-fly and generates a functional tabular report."""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} does not exist.")
        return

    data = json.loads(input_path.read_text(encoding="utf-8"))
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Functional LCR Annotation Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f8fafc; color: #334155; margin: 0; padding: 24px; font-size: 13px; }
        .container { max-width: 1500px; margin: 0 auto; }
        h1 { font-size: 24px; color: #1e293b; margin-bottom: 4px; }
        .subtitle { color: #64748b; margin-bottom: 24px; }
        table { width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
        th { background-color: #1e293b; color: #ffffff; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
        tr:hover td { background-color: #f8fafc; }
        .protein-id { font-family: monospace; font-weight: bold; color: #2563eb; text-decoration: none; }
        .protein-id:hover { text-decoration: underline; }
        .gene-name { font-weight: bold; color: #1e293b; }
        .evidence-quote { font-style: italic; color: #475569; margin: 0; border-left: 3px solid #cbd5e1; padding-left: 8px; }
        .badge { background: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-family: monospace; }
        .source-tag { font-size: 11px; color: #64748b; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Functional LCR Annotation Report</h1>
        <p class="subtitle">Curated Low-Complexity Regions integrated with UniProt metadata</p>
        
        <table>
            <thead>
                <tr>
                    <th style="width: 11%;">UniProt / Gene</th>
                    <th style="width: 15%;">Name / Organism</th>
                    <th style="width: 6%;">Length</th>
                    <th style="width: 8%;">Annot. Coords</th>
                    <th style="width: 8%;">PlatoLoCo Coords</th>
                    <th style="width: 14%;">Function / Binding</th>
                    <th style="width: 26%;">Evidence</th>
                    <th style="width: 12%;">Source File</th>
                </tr>
            </thead>
            <tbody>
"""

    for item in data:
        protein_name = item.get("protein_name", "")
        organism = item.get("organism", "")
        
        print(f"Fetching UniProt data for: {protein_name} ({organism})...")
        uni_data = fetch_uniprot_metadata(protein_name, organism)
        
        uniprot_id = uni_data.get("uniprot_id", "N/A")
        gene_name = uni_data.get("gene_name", protein_name)
        full_name = uni_data.get("full_name", protein_name)
        length = uni_data.get("length", "N/A")
        
        uniprot_link = f"https://www.uniprot.org/uniprotkb/{uniprot_id}" if uniprot_id != "N/A" else "#"
        
        html_content += f"""
                <tr>
                    <td>
                        <a href="{uniprot_link}" target="_blank" class="protein-id">{uniprot_id}</a><br>
                        <span class="gene-name">{gene_name}</span>
                    </td>
                    <td>
                        <strong>{full_name}</strong><br>
                        <i style="color: #64748b;">{organism}</i>
                    </td>
                    <td>{length}</td>
                    <td><span class="badge">{item.get('start_of_annotation')} - {item.get('end_of_annotation')}</span></td>
                    <td><span style="color: #94a3b8;">Pending...</span></td>
                    <td>
                        <strong>{item.get('binding_target', 'Unspecified')}</strong><br>
                        <span style="font-size: 12px; color: #475569;">{item.get('proposed_function', '')}</span>
                    </td>
                    <td><blockquote class="evidence-quote">"{item.get('evidence', '')}"</blockquote></td>
                    <td>{item.get('file', '')}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Success! Report saved to: {output_html}")


if __name__ == "__main__":
    generate_html_report()
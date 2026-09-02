import json
import re
from pathlib import Path


def extract_doi_from_text(text: str) -> str:
    """Extracts DOI pattern from text or filename."""
    match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text)
    return match.group(0) if match else ""


def generate_html_report(
    input_file: str = "data/processed/final_results.jsonl",
    output_html: str = "data/processed/lcr_biocuration_report.html"
):
    """Generates an interactive HTML biocuration dashboard with UniProt and DOI links."""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} does not exist.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    verified = []
    qualitative = []

    for entry in lines:
        fname = entry.get("file", "Unspecified")
        entry_doi = entry.get("doi", extract_doi_from_text(fname))

        for ann in entry.get("annotations", []):
            p_name = ann.get("protein_name", "").strip()
            start = str(ann.get("start_of_annotation", "")).strip()
            end = str(ann.get("end_of_annotation", "")).strip()
            ann_doi = ann.get("doi", entry_doi)

            item = {
                "file": fname,
                "protein": p_name,
                "organism": ann.get("organism", "Unspecified"),
                "binding_target": ann.get("binding_target", "Unspecified"),
                "evidence": ann.get("evidence", ""),
                "note": ann.get("curator_note", ""),
                "doi": ann_doi
            }

            if start.isdigit() and end.isdigit():
                item["start"] = start
                item["end"] = end
                item["function"] = ann.get("proposed_function", "N/A")
                verified.append(item)
            else:
                qualitative.append(item)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LCR / LCD Binding Biocuration Report</title>
    <style>
        :root {{
            --primary: #1e293b;
            --accent-blue: #2563eb;
            --accent-amber: #d97706;
            --bg-main: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #334155;
            --border-color: #e2e8f0;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: var(--bg-main); color: var(--text-main); margin: 0; padding: 32px 24px; }}
        .container {{ max-width: 1380px; margin: 0 auto; }}
        header {{ margin-bottom: 24px; border-bottom: 2px solid var(--border-color); padding-bottom: 16px; }}
        h1 {{ font-size: 26px; color: var(--primary); margin: 0 0 8px 0; }}
        .subtitle {{ color: #64748b; font-size: 14px; margin: 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .stat-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .stat-value {{ font-size: 28px; font-weight: 700; color: var(--primary); margin-bottom: 4px; }}
        .stat-card.verified .stat-value {{ color: var(--accent-blue); }}
        .stat-card.qualitative .stat-value {{ color: var(--accent-amber); }}
        .stat-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
        .section-title {{ font-size: 18px; font-weight: 600; margin: 32px 0 16px 0; display: flex; align-items: center; gap: 10px; color: var(--primary); }}
        .badge {{ font-size: 12px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }}
        .badge-blue {{ background: #dbeafe; color: #1e40af; }}
        .badge-amber {{ background: #fef3c7; color: #92400e; }}
        .badge-target {{ background: #f3e8ff; color: #6b21a8; font-weight: 700; padding: 3px 8px; border-radius: 6px; font-size: 12px; display: inline-block; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color); box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 32px; font-size: 14px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border-color); vertical-align: top; }}
        th {{ background-color: #f1f5f9; color: var(--primary); font-weight: 600; font-size: 13px; }}
        tr:hover td {{ background-color: #f8fafc; }}
        .protein-name {{ font-weight: 700; color: var(--primary); font-family: monospace; font-size: 15px; }}
        .file-tag {{ font-size: 11px; color: #64748b; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px; }}
        .evidence-quote {{ font-style: italic; color: #475569; border-left: 3px solid #cbd5e1; padding-left: 8px; margin: 0; font-size: 13px; }}
        .btn-link {{ color: var(--accent-blue); text-decoration: none; font-weight: 600; font-size: 12px; background: #eff6ff; padding: 5px 9px; border-radius: 4px; display: inline-block; border: 1px solid #bfdbfe; white-space: nowrap; margin-bottom: 4px; }}
        .btn-link:hover {{ background: #dbeafe; }}
        .btn-doi {{ color: #475569; background: #f1f5f9; border-color: #cbd5e1; }}
        .btn-doi:hover {{ background: #e2e8f0; }}
        .actions-cell {{ display: flex; flex-direction: column; gap: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LCR / LCD Binding Biocuration Report</h1>
            <p class="subtitle">Automated extraction of Low-Complexity Regions engaging in molecular binding interactions</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(lines)}</div>
                <div class="stat-label">Analyzed Papers</div>
            </div>
            <div class="stat-card verified">
                <div class="stat-value">{len(verified)}</div>
                <div class="stat-label">Verified LCRs (Exact Boundaries)</div>
            </div>
            <div class="stat-card qualitative">
                <div class="stat-value">{len(qualitative)}</div>
                <div class="stat-label">Qualitative Mentions (To Map)</div>
            </div>
        </div>

        <div class="section-title">
            <span>Verified Binding LCR Regions (Numerical Coordinates)</span>
            <span class="badge badge-blue">{len(verified)}</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th style="width: 16%;">Protein / Source File</th>
                    <th style="width: 10%;">Organism</th>
                    <th style="width: 10%;">Residues</th>
                    <th style="width: 12%;">Binding Target</th>
                    <th style="width: 22%;">Proposed Function</th>
                    <th style="width: 20%;">Verbatim Evidence</th>
                    <th style="width: 10%;">Links</th>
                </tr>
            </thead>
            <tbody>
"""

    for v in verified:
        doi_html = f'<a href="https://doi.org/{v["doi"]}" target="_blank" class="btn-link btn-doi">DOI</a>' if v["doi"] else ''
        uniprot_url = f"https://www.uniprot.org/uniprotkb?query={v['protein']}+{v['organism']}"

        html_content += f"""
                <tr>
                    <td>
                        <div class="protein-name">{v['protein']}</div>
                        <div class="file-tag">{v['file']}</div>
                    </td>
                    <td>{v['organism']}</td>
                    <td><strong>{v['start']} - {v['end']}</strong></td>
                    <td><span class="badge-target">{v['binding_target']}</span></td>
                    <td>{v['function']}</td>
                    <td><blockquote class="evidence-quote">"{v['evidence']}"</blockquote></td>
                    <td class="actions-cell">
                        <a href="{uniprot_url}" target="_blank" class="btn-link">UniProt</a>
                        {doi_html}
                    </td>
                </tr>
        """

    html_content += f"""
            </tbody>
        </table>

        <div class="section-title">
            <span>Qualitative Binding Mentions (Require UniProt Mapping)</span>
            <span class="badge badge-amber">{len(qualitative)}</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th style="width: 16%;">Protein / Source File</th>
                    <th style="width: 10%;">Organism</th>
                    <th style="width: 12%;">Binding Target</th>
                    <th style="width: 30%;">Verbatim Evidence</th>
                    <th style="width: 22%;">Curator Note</th>
                    <th style="width: 10%;">Links</th>
                </tr>
            </thead>
            <tbody>
"""

    for q in qualitative:
        doi_html = f'<a href="https://doi.org/{q["doi"]}" target="_blank" class="btn-link btn-doi">DOI</a>' if q["doi"] else ''
        uniprot_url = f"https://www.uniprot.org/uniprotkb?query={q['protein']}+{q['organism']}"

        html_content += f"""
                <tr>
                    <td>
                        <div class="protein-name">{q['protein']}</div>
                        <div class="file-tag">{q['file']}</div>
                    </td>
                    <td>{q['organism']}</td>
                    <td><span class="badge-target">{q['binding_target']}</span></td>
                    <td><blockquote class="evidence-quote">"{q['evidence']}"</blockquote></td>
                    <td>{q['note']}</td>
                    <td class="actions-cell">
                        <a href="{uniprot_url}" target="_blank" class="btn-link">UniProt</a>
                        {doi_html}
                    </td>
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
    print(f"HTML Report generated at: {output_html}")


if __name__ == "__main__":
    generate_html_report()
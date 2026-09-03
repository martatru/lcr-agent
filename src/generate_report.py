"""
LCR Biocuration HTML Report Generator.

Integrates curated Low-Complexity Region (LCR) records with UniProt metadata
and multi-track sequence visualizers utilizing supported PlaToLoCo predictors.
"""

import json
from pathlib import Path
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional

import requests

PLATOLOCO_API_URL = "http://127.0.0.1:5002/restapi"


def load_input_data(input_path: Path) -> List[Dict[str, Any]]:
    """Load JSON or JSONL data and flatten annotation lists if present."""
    if not input_path.exists():
        return []

    raw_content = input_path.read_text(encoding="utf-8").strip()
    if not raw_content:
        return []

    records: List[Dict[str, Any]] = []

    def _process_item(item: Dict[str, Any]) -> None:
        file_name = item.get("file", "N/A")
        if "annotations" in item and isinstance(item["annotations"], list):
            for annot in item["annotations"]:
                if isinstance(annot, dict):
                    annot_copy = dict(annot)
                    annot_copy.setdefault("file", file_name)
                    records.append(annot_copy)
        else:
            records.append(item)

    if input_path.suffix.lower() == ".jsonl" or (
        "\n" in raw_content and not raw_content.startswith("[")
    ):
        for line in raw_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    _process_item(parsed)
            except json.JSONDecodeError:
                continue
    else:
        try:
            parsed_data = json.loads(raw_content)
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    if isinstance(item, dict):
                        _process_item(item)
            elif isinstance(parsed_data, dict):
                _process_item(parsed_data)
        except json.JSONDecodeError:
            pass

    return records


def fetch_uniprot_metadata(protein_name: str, organism: str) -> Dict[str, Any]:
    """Fetch protein metadata, length, sequence, and GO terms from UniProt API."""
    query = f"({protein_name}) AND (organism_name:{organism})"
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        f"query={urllib.parse.quote(query)}&format=json&size=1"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.json().get("results"):
            res = response.json()["results"][0]

            go_terms = [
                f"{ref.get('id')} ({ref.get('properties', [{}])[0].get('value', 'GO')})"
                if ref.get("properties")
                else ref.get("id", "")
                for ref in res.get("uniProtKBCrossReferences", [])
                if ref.get("database") == "GO"
            ]

            genes = res.get("genes", [{}])
            gene_name = (
                genes[0].get("geneName", {}).get("value", protein_name)
                if genes
                else protein_name
            )
            full_name = (
                res.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", protein_name)
            )

            return {
                "uniprot_id": res.get("primaryAccession", "N/A"),
                "gene_name": gene_name,
                "full_name": full_name,
                "length": res.get("sequence", {}).get("length", 0),
                "sequence": res.get("sequence", {}).get("value", ""),
                "go_terms": go_terms[:3],
            }
    except Exception as error:
        print(f"UniProt query error for {protein_name}: {error}")

    return {
        "uniprot_id": "N/A",
        "gene_name": protein_name,
        "full_name": protein_name,
        "length": 0,
        "sequence": "",
        "go_terms": [],
    }


def query_platoloco(
    sequence: str, header: str = "seq"
) -> Dict[str, List[Dict[str, int]]]:
    """Submit sequence to PlaToLoCo API requesting supported core prediction methods[cite: 2]."""
    method_results: Dict[str, List[Dict[str, int]]] = {
        "SEG_intermediate": [],
        "SEG_strict": [],
        "CAST": [],
        "fLPS_strict": [],
    }
    if not sequence:
        return method_results

    unique_job_id = f"{header}_{uuid.uuid4().hex[:6]}"
    flps_params = {
        "min_tract_len": 15,
        "max_tract_len": 500,
        "pval": 0.001,
        "regions": {"single": True, "multiple": True, "whole": False},
    }

    # Restrict to verified core callers configuration[cite: 2]
    payload = {
        "name": unique_job_id,
        "sequences": f">{header}\n{sequence.strip()}\n",
        "methods": {
            "seg_default": False,
            "seg_intermediate": True,
            "seg_strict": True,
            "cast": True,
            "flps": False,
            "flps_strict": True,
            "simple": False,
            "gbsc": False,
        },
        "enrichment": {
            "pfam": False,
            "phobius": False,
            "aafrequency": False,
        },
        "params": {
            "seg_strict": {},
            "seg_intermediate": {"window": 15, "k1": 1.9, "k2": 2.5},
            "cast": {"threshold": 40, "matrix": 1},
            "flps_strict": flps_params,
        },
    }

    canonical_map = {
        "seg_intermediate": "SEG_intermediate",
        "SEG_intermediate": "SEG_intermediate",
        "seg_strict": "SEG_strict",
        "SEG_strict": "SEG_strict",
        "cast": "CAST",
        "CAST": "CAST",
        "flps_strict": "fLPS_strict",
        "fLPS_strict": "fLPS_strict",
    }

    try:
        res = requests.put(f"{PLATOLOCO_API_URL}/query", json=payload, timeout=10)
        if res.status_code != 200:
            return method_results

        token = res.json().get("token")
        if not token:
            return method_results

        for _ in range(30):
            status_res = requests.get(
                f"{PLATOLOCO_API_URL}/job/{token}", timeout=5
            )
            if status_res.status_code == 200:
                st = status_res.json().get("status")
                if st == "FINISHED":
                    break
                if st == "ERROR":
                    print(f"PlaToLoCo execution error for token {token}")
                    return method_results
            time.sleep(1)

        list_res = requests.get(f"{PLATOLOCO_API_URL}/proteins/{token}", timeout=10)
        if list_res.status_code != 200:
            return method_results

        proteins = list_res.json().get("proteins", [])
        if not proteins:
            return method_results

        prot_summary = proteins[0]
        p_internal_id = prot_summary.get("id")

        if p_internal_id is not None:
            try:
                details_res = requests.get(
                    f"{PLATOLOCO_API_URL}/proteins/{token}/{p_internal_id}",
                    timeout=10,
                )
                if details_res.status_code == 200:
                    wrapper_items = (
                        details_res.json()
                        .get("data", {})
                        .get("wrapper", [])
                    )
                    for item in wrapper_items:
                        raw_m = item.get("method", "")
                        std_m = canonical_map.get(raw_m) or canonical_map.get(
                            raw_m.lower()
                        )
                        if std_m in method_results:
                            for reg in item.get("regions", []):
                                try:
                                    method_results[std_m].append({
                                        "start": int(reg["beg"]),
                                        "end": int(reg["end"]),
                                    })
                                except (KeyError, ValueError, TypeError):
                                    pass
            except Exception as err:
                print(f"Detail parsing error: {err}")

        for key, val in prot_summary.items():
            std_m = canonical_map.get(key) or canonical_map.get(key.lower())
            if std_m in method_results and isinstance(val, list) and val:
                if not method_results[std_m]:
                    for reg in val:
                        if isinstance(reg, list) and len(reg) == 2:
                            try:
                                method_results[std_m].append({
                                    "start": int(reg[0]),
                                    "end": int(reg[1]),
                                })
                            except (ValueError, TypeError):
                                pass

        for std_m in method_results:
            seen = set()
            unique = []
            for reg in sorted(
                method_results[std_m], key=lambda x: (x["start"], x["end"])
            ):
                pair = (reg["start"], reg["end"])
                if pair not in seen:
                    seen.add(pair)
                    unique.append(reg)
            method_results[std_m] = unique

        return method_results

    except Exception as err:
        print(f"PlaToLoCo communication error: {err}")
        return method_results


def generate_platoloco_style_svg(
    seq_length: int,
    annot_start: Optional[int],
    annot_end: Optional[int],
    platoloco_methods: Dict[str, List[Dict[str, int]]],
) -> str:
    """Generate a multi-track SVG map matching PlaToLoCo's native UI visual style."""
    if not isinstance(seq_length, int) or seq_length <= 0:
        return '<span style="color: #94a3b8; font-size: 11px;">Sequence length unavailable</span>'

    annot_regions = []
    if (
        annot_start is not None
        and annot_end is not None
        and annot_start <= annot_end
    ):
        annot_regions.append({"start": annot_start, "end": annot_end})

    tracks = [
        {"label": "Annotated LCR", "color": "#f97316", "regions": annot_regions},
        {
            "label": "SEG-intermediate",
            "color": "#c026d3",
            "regions": platoloco_methods.get("SEG_intermediate", []),
        },
        {
            "label": "SEG-strict",
            "color": "#a21caf",
            "regions": platoloco_methods.get("SEG_strict", []),
        },
        {"label": "CAST", "color": "#a21caf", "regions": platoloco_methods.get("CAST", [])},
        {
            "label": "fLPS-strict",
            "color": "#f43f5e",
            "regions": platoloco_methods.get("fLPS_strict", []),
        },
    ]

    label_width = 150
    track_area_width = 850
    total_width = label_width + track_area_width + 30
    row_height = 24
    top_offset = 12
    ruler_height = 30
    total_height = top_offset + (len(tracks) * row_height) + ruler_height

    svg_elements = []

    for idx, track in enumerate(tracks):
        y_base = top_offset + (idx * row_height) + 12

        tag_x = 10
        tag_w = label_width - 25
        tag_h = 16
        tag_y = y_base - 8
        chevron_w = 6

        tag_path = (
            f"M {tag_x} {tag_y} "
            f"L {tag_x + tag_w - chevron_w} {tag_y} "
            f"L {tag_x + tag_w} {tag_y + (tag_h / 2)} "
            f"L {tag_x + tag_w - chevron_w} {tag_y + tag_h} "
            f"L {tag_x} {tag_y + tag_h} Z"
        )

        svg_elements.append(f'<path d="{tag_path}" fill="#e2e8f0"/>')

        svg_elements.append(
            f'<text x="{tag_x + 8}" y="{y_base + 3}" fill="#475569" '
            f'font-size="10" font-weight="600" font-family="sans-serif">'
            f'{track["label"]}</text>'
        )

        svg_elements.append(
            f'<line x1="{label_width}" y1="{y_base}" x2="{label_width + track_area_width}" '
            f'y2="{y_base}" stroke="#e2e8f0" stroke-width="1.5"/>'
        )

        for reg in track["regions"]:
            p_start = reg.get("start", 1)
            p_end = reg.get("end", 1)
            x_pos = label_width + (p_start / seq_length) * track_area_width
            rect_w = max(((p_end - p_start) / seq_length) * track_area_width, 4)

            svg_elements.append(
                f'<rect x="{x_pos:.1f}" y="{y_base - 5}" width="{rect_w:.1f}" height="10" '
                f'fill="{track["color"]}" rx="1">'
                f'<title>{track["label"]}: {p_start}-{p_end}</title></rect>'
            )

    ruler_y = top_offset + (len(tracks) * row_height) + 6
    svg_elements.append(
        f'<line x1="{label_width}" y1="{ruler_y}" x2="{label_width + track_area_width}" '
        f'y2="{ruler_y}" stroke="#334155" stroke-width="1.5"/>'
    )

    tick_step = 50 if seq_length <= 350 else (100 if seq_length <= 1000 else 200)
    curr_tick = 0
    while curr_tick <= seq_length:
        x_tick = label_width + (curr_tick / seq_length) * track_area_width
        svg_elements.append(
            f'<line x1="{x_tick:.1f}" y1="{ruler_y}" x2="{x_tick:.1f}" y2="{ruler_y + 5}" '
            f'stroke="#334155" stroke-width="1.5"/>'
        )
        svg_elements.append(
            f'<text x="{x_tick:.1f}" y="{ruler_y + 18}" fill="#475569" font-size="10" '
            f'text-anchor="middle" font-family="sans-serif">{curr_tick}</text>'
        )
        curr_tick += tick_step

    return (
        f'<svg width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}" '
        f'style="background: #ffffff; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; display: block; margin-left: 0;">'
        f'{"".join(svg_elements)}</svg>'
    )


def parse_coord(value: Any) -> Optional[int]:
    """Safely convert coordinate values to integers or return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def render_record_rows(item: Dict[str, Any]) -> str:
    """Render table row and subrow visual map for a single dataset record."""
    protein_name = item.get("protein_name") or "Unknown"
    organism = item.get("organism") or "Unspecified"

    print(f"Fetching UniProt data for: {protein_name} ({organism})...")
    uni_data = fetch_uniprot_metadata(protein_name, organism)

    uniprot_id = uni_data["uniprot_id"]
    gene_name = uni_data["gene_name"]
    full_name = uni_data["full_name"]
    length = uni_data["length"]
    sequence = uni_data["sequence"]
    go_terms = uni_data.get("go_terms", [])

    platoloco_methods = {}
    if sequence:
        print(f"Querying PlaToLoCo for {protein_name}...")
        platoloco_methods = query_platoloco(sequence, header=uniprot_id)

    lcr_type_val = (
        item.get("lcr_type")
        or item.get("binding_target")
        or item.get("proposed_function")
        or item.get("annotation_category")
        or "Unspecified"
    )

    annot_start = parse_coord(item.get("start_of_annotation"))
    annot_end = parse_coord(item.get("end_of_annotation"))

    start_annot_str = (
        str(annot_start)
        if annot_start is not None
        else item.get("start_of_annotation", "Unspecified")
    )
    end_annot_str = (
        str(annot_end)
        if annot_end is not None
        else item.get("end_of_annotation", "Unspecified")
    )

    svg_track = generate_platoloco_style_svg(
        length,
        annot_start,
        annot_end,
        platoloco_methods,
    )

    uniprot_link = (
        f"https://www.uniprot.org/uniprotkb/{uniprot_id}"
        if uniprot_id != "N/A"
        else "#"
    )

    evidence_text = item.get("evidence") or "No evidence statement provided."
    source_id = (
        item.get("source_id") or item.get("doi") or item.get("file") or "N/A"
    )
    category = (
        item.get("annotation_category")
        or item.get("proposed_function")
        or item.get("binding_target")
        or "Unspecified"
    )
    go_ontology_str = "<br>".join(go_terms) if go_terms else "N/A"

    return f"""
            <tr>
                <td><a href="{uniprot_link}" target="_blank" class="protein-id">{uniprot_id}</a></td>
                <td><strong>{gene_name}</strong></td>
                <td style="max-width: 180px;">{full_name}</td>
                <td></td>
                <td></td>
                <td>{length if length > 0 else 'N/A'}</td>
                <td><span class="badge-type">{lcr_type_val}</span></td>
                <td><i>{organism}</i></td>
                <td class="col-source"><blockquote class="evidence-quote">"{evidence_text}"</blockquote></td>
                <td><code>{source_id}</code></td>
                <td><span class="badge-annot">{start_annot_str}</span></td>
                <td><span class="badge-annot">{end_annot_str}</span></td>
                <td style="max-width: 180px;"><strong>{category}</strong></td>
                <td style="max-width: 200px;">{go_ontology_str}</td>
            </tr>
            <tr class="subrow">
                <td colspan="14" class="viz-container">
                    <div class="viz-header">
                        <span>Sequence details ({length} aa)</span>
                    </div>
                    {svg_track}
                </td>
            </tr>
"""


def generate_html_report(
    input_file: str = "data/processed/verified_lcrs.json",
    output_html: str = "data/processed/lcr_biocuration_report.html",
) -> None:
    """Generate an HTML biocuration report grouping unspecified ranges at the bottom[cite: 3]."""
    input_path = Path(input_file)
    if not input_path.exists():
        alt_path = Path("data/processed/final_results.jsonl")
        if alt_path.exists():
            input_path = alt_path
        else:
            print(f"Error: Input file {input_file} does not exist.")
            return

    data = load_input_data(input_path)
    if not data:
        print(f"Warning: No valid records found in {input_path}.")
        return

    specified_records = []
    unspecified_records = []

    for item in data:
        st_val = parse_coord(item.get("start_of_annotation"))
        end_val = parse_coord(item.get("end_of_annotation"))
        if st_val is not None and end_val is not None:
            specified_records.append(item)
        else:
            unspecified_records.append(item)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Functional LCR Annotation Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f8fafc; color: #334155; margin: 0; padding: 16px; font-size: 12px; }
        .container { width: 100%; max-width: 1920px; margin: 0 auto; overflow-x: auto; }
        h1 { font-size: 20px; color: #1e293b; margin-bottom: 2px; }
        .subtitle { color: #64748b; margin-bottom: 16px; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 6px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); table-layout: auto; }
        th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: top; white-space: normal; word-wrap: break-word; }
        th { background-color: #1e293b; color: #ffffff; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap; }
        tr:hover td { background-color: #f8fafc; }
        .protein-id { font-family: monospace; font-weight: bold; color: #2563eb; text-decoration: none; }
        .col-source { min-width: 450px; max-width: 650px; }
        .evidence-quote { font-style: italic; color: #475569; margin: 0; border-left: 3px solid #cbd5e1; padding-left: 8px; line-height: 1.4; }
        .badge-annot { background: #ffedd5; color: #c2410c; padding: 2px 5px; border-radius: 4px; font-weight: bold; font-family: monospace; }
        .badge-type { background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; display: inline-block; }
        .subrow { background-color: #f8fafc; border-bottom: 2px solid #cbd5e1; }
        .viz-container { padding: 12px; text-align: left; }
        .viz-header { font-size: 12px; font-weight: bold; color: #1e293b; margin-bottom: 6px; text-align: left; }
        .category-header-row td { background-color: #334155; color: #ffffff; font-weight: 700; font-size: 12px; padding: 10px 12px; letter-spacing: 0.5px; text-transform: uppercase; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Functional LCR Annotation Report</h1>
        <p class="subtitle">Structured low-complexity region dataset integrated with UniProt and PlaToLoCo predictions</p>
        
        <table>
            <thead>
                <tr>
                    <th>UniprotID</th>
                    <th>Gene name</th>
                    <th>Name</th>
                    <th>Start of LCR</th>
                    <th>End of LCR</th>
                    <th>Protein length</th>
                    <th>LCR type</th>
                    <th>Organism</th>
                    <th class="col-source">Source</th>
                    <th>Source ID</th>
                    <th>Start of annotation</th>
                    <th>End of annotation</th>
                    <th>Annotation Category</th>
                    <th>Gene Ontology of category</th>
                </tr>
            </thead>
            <tbody>
"""

    for item in specified_records:
        html_content += render_record_rows(item)

    if unspecified_records:
        html_content += """
                <tr class="category-header-row">
                    <td colspan="14">LCRs with Unspecified Range</td>
                </tr>
"""
        for item in unspecified_records:
            html_content += render_record_rows(item)

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
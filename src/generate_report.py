"""
LCR Biocuration HTML Report Generator.

Fetches curated Low-Complexity Region (LCR) records directly from the SQLite database
and integrates them with UniProt metadata and PlaToLoCo sequence visualizers.
"""

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Adjust Python search path to resolve relative module imports
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PLATOLOCO_API_URL = "http://127.0.0.1:5002/restapi"
DB_PATH = PROJECT_ROOT / "data" / "lcr_annotations.db"

# Global caches to avoid repeating identical API calls
UNIPROT_CACHE: Dict[str, Dict[str, Any]] = {}
PLATOLOCO_CACHE: Dict[str, Dict[str, List[Dict[str, int]]]] = {}

# Single shared session reuses TCP connections across requests
HTTP_SESSION = requests.Session()


def fetch_records_from_db(paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch LCR annotation records directly from SQLite database."""
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}")
        return []

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT 
                a.paper_id,
                a.doi,
                COALESCE(p.file_name, 'N/A') AS file,
                a.protein_name,
                a.organism,
                a.start_pos AS start_of_annotation,
                a.end_pos AS end_of_annotation,
                a.binding_target,
                a.proposed_function,
                a.evidence,
                a.evidence_verified,
                a.curation_status
            FROM lcr_annotations a
            LEFT JOIN processed_papers p ON a.paper_id = p.paper_id
        """
        params: List[Any] = []
        if paper_id:
            query += " WHERE a.paper_id = ?"
            params.append(paper_id)

        query += " ORDER BY a.id ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        records = []
        for row in rows:
            record = dict(row)
            record["evidence_verified"] = bool(record.get("evidence_verified"))
            records.append(record)
        return records


def parse_coord(value: Any) -> Optional[int]:
    """Safely convert coordinate values to integers or return None."""
    if value is None:
        return None
    try:
        val_str = str(value).strip()
        if val_str.isdigit():
            return int(val_str)
        return None
    except (ValueError, TypeError):
        return None


def fetch_uniprot_metadata(protein_name: str, organism: str) -> Dict[str, Any]:
    """Fetch protein metadata, length, sequence, and GO terms from UniProt API with caching."""
    clean_protein = re.sub(r"[^\w\s-]", "", protein_name).strip()
    clean_organism = re.sub(r"[^\w\s-]", "", organism).strip()

    cache_key = f"{clean_protein}_{clean_organism}"
    if cache_key in UNIPROT_CACHE:
        return UNIPROT_CACHE[cache_key]

    search_queries = [
        f'(gene:{clean_protein} OR gene_exact:{clean_protein} OR protein_name:{clean_protein}) AND (organism_name:"{clean_organism}")',
        f'("{clean_protein}") AND (organism_name:"{clean_organism}")',
        f'{clean_protein} AND (organism_name:"{clean_organism}")',
        f"(gene:{clean_protein} OR protein_name:{clean_protein})",
    ]

    for query in search_queries:
        url = (
            "https://rest.uniprot.org/uniprotkb/search?"
            f"query={urllib.parse.quote(query)}&format=json&size=1"
        )

        try:
            response = HTTP_SESSION.get(url, timeout=10)
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

                result = {
                    "uniprot_id": res.get("primaryAccession", "N/A"),
                    "gene_name": gene_name,
                    "full_name": full_name,
                    "length": res.get("sequence", {}).get("length", 0),
                    "sequence": res.get("sequence", {}).get("value", ""),
                    "go_terms": go_terms[:3],
                }
                UNIPROT_CACHE[cache_key] = result
                return result
        except Exception as error:
            print(f"UniProt query error for '{protein_name}': {error}")
            continue

    fallback_result = {
        "uniprot_id": "N/A",
        "gene_name": protein_name,
        "full_name": protein_name,
        "length": 0,
        "sequence": "",
        "go_terms": [],
    }
    UNIPROT_CACHE[cache_key] = fallback_result
    return fallback_result


def query_single_platoloco_method(
    sequence: str, method_name: str, header: str = "seq"
) -> List[Dict[str, int]]:
    """Query a single PlaToLoCo method independently to isolate execution errors."""
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

    gbsc_params = {"score": 0, "distance": 0}

    methods_flag = {
        "seg_default": False,
        "seg_intermediate": False,
        "seg_strict": False,
        "cast": False,
        "flps": False,
        "flps_strict": False,
        "simple": False,
        "gbsc": False,
    }
    methods_flag[method_name] = True

    payload = {
        "name": f"{header}_{method_name}_{uuid.uuid4().hex[:4]}",
        "sequences": f">{header}\n{sequence.strip()}\n",
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

    regions_list: List[Dict[str, int]] = []

    try:
        res = HTTP_SESSION.put(f"{PLATOLOCO_API_URL}/query", json=payload, timeout=12)
        if res.status_code != 200:
            return regions_list

        token = res.json().get("token")
        if not token:
            return regions_list

        for _ in range(25):
            status_res = HTTP_SESSION.get(f"{PLATOLOCO_API_URL}/job/{token}", timeout=5)
            if status_res.status_code == 200:
                st = status_res.json().get("status")
                if st == "FINISHED":
                    break
                if st == "ERROR":
                    return regions_list
            time.sleep(0.5)

        list_res = HTTP_SESSION.get(f"{PLATOLOCO_API_URL}/proteins/{token}", timeout=10)
        if list_res.status_code != 200:
            return regions_list

        proteins = list_res.json().get("proteins", [])
        if not proteins:
            return regions_list

        prot_summary = proteins[0]
        p_internal_id = prot_summary.get("id")

        if p_internal_id is not None:
            details_res = HTTP_SESSION.get(
                f"{PLATOLOCO_API_URL}/proteins/{token}/{p_internal_id}",
                timeout=10,
            )
            if details_res.status_code == 200:
                wrapper_items = (
                    details_res.json().get("data", {}).get("wrapper", [])
                )
                for item in wrapper_items:
                    for reg in item.get("regions", []):
                        try:
                            regions_list.append({
                                "start": int(reg["beg"]),
                                "end": int(reg["end"]),
                            })
                        except (KeyError, ValueError, TypeError):
                            pass

    except Exception as err:
        print(f"PlaToLoCo method '{method_name}' query error: {err}")

    return regions_list


def query_platoloco(
    sequence: str, header: str = "seq"
) -> Dict[str, List[Dict[str, int]]]:
    """Query all 8 PlaToLoCo predictors using isolated requests and caching."""
    if not sequence:
        return {}

    if sequence in PLATOLOCO_CACHE:
        return PLATOLOCO_CACHE[sequence]

    method_results: Dict[str, List[Dict[str, int]]] = {
        "SEG": [],
        "SEG_intermediate": [],
        "SEG_strict": [],
        "CAST": [],
        "fLPS": [],
        "fLPS_strict": [],
        "SIMPLE": [],
        "GBSC": [],
    }

    method_key_map = {
        "seg_default": "SEG",
        "seg_intermediate": "SEG_intermediate",
        "seg_strict": "SEG_strict",
        "cast": "CAST",
        "flps": "fLPS",
        "flps_strict": "fLPS_strict",
        "simple": "SIMPLE",
        "gbsc": "GBSC",
    }

    for req_key, canonical_key in method_key_map.items():
        regs = query_single_platoloco_method(sequence, req_key, header)
        method_results[canonical_key] = regs

    PLATOLOCO_CACHE[sequence] = method_results
    return method_results


def merge_regions(regions: List[Dict[str, int]], max_gap: int = 4) -> List[Dict[str, int]]:
    """Merge overlapping or closely neighboring coordinate regions."""
    if not regions:
        return []
    sorted_regs = sorted(regions, key=lambda x: x.get("start", 0))
    merged = [dict(sorted_regs[0])]
    for current in sorted_regs[1:]:
        prev = merged[-1]
        if current.get("start", 0) <= prev.get("end", 0) + max_gap:
            prev["end"] = max(prev.get("end", 0), current.get("end", 0))
        else:
            merged.append(dict(current))
    return merged


def generate_platoloco_style_svg(
    seq_length: int,
    annot_start: Optional[int],
    annot_end: Optional[int],
    platoloco_methods: Dict[str, List[Dict[str, int]]],
    uniprot_id: str = "protein",
) -> str:
    """Generate multi-track SVG visualizer with merged regions and hover tooltips."""
    if not isinstance(seq_length, int) or seq_length <= 0:
        return '<span style="color: #64748b; font-size: 11px;">Sequence length unavailable</span>'

    annot_regions = []
    if annot_start is not None and annot_end is not None and annot_start <= annot_end:
        annot_regions.append({"start": annot_start, "end": annot_end})

    tracks = [
        {"label": "Annotated Sequence", "color": "#f97316", "regions": annot_regions},
        {"label": "SEG", "color": "#d946ef", "regions": platoloco_methods.get("SEG", [])},
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
        {"label": "fLPS", "color": "#db2777", "regions": platoloco_methods.get("fLPS", [])},
        {
            "label": "fLPS-strict",
            "color": "#f43f5e",
            "regions": platoloco_methods.get("fLPS_strict", []),
        },
        {"label": "SIMPLE", "color": "#64748b", "regions": platoloco_methods.get("SIMPLE", [])},
        {"label": "GBSC", "color": "#475569", "regions": platoloco_methods.get("GBSC", [])},
    ]

    label_width = 150
    track_area_width = 850
    total_width = label_width + track_area_width + 40
    row_height = 28
    top_offset = 36
    ruler_height = 30
    total_height = top_offset + (len(tracks) * row_height) + ruler_height + 12

    svg_elements = [
        f'<rect x="0" y="0" width="{total_width}" height="{total_height}" fill="#ffffff" rx="6" stroke="#e2e8f0" stroke-width="1"/>',
        f'<text x="16" y="24" fill="#0f172a" font-size="12" font-weight="700" font-family="sans-serif">Sequence details ({seq_length} aa)</text>',
    ]

    for idx, track in enumerate(tracks):
        y_base = top_offset + (idx * row_height) + 14

        tag_x, tag_w, tag_h, tag_y, chevron_w = 16, label_width - 25, 16, y_base - 8, 6
        tag_path = (
            f"M {tag_x} {tag_y} "
            f"L {tag_x + tag_w - chevron_w} {tag_y} "
            f"L {tag_x + tag_w} {tag_y + (tag_h / 2)} "
            f"L {tag_x + tag_w - chevron_w} {tag_y + tag_h} "
            f"L {tag_x} {tag_y + tag_h} Z"
        )

        svg_elements.append(f'<path d="{tag_path}" fill="#f1f5f9"/>')
        svg_elements.append(
            f'<text x="{tag_x + 8}" y="{y_base + 3}" fill="#475569" font-size="10" font-weight="600" font-family="sans-serif">{track["label"]}</text>'
        )

        start_x_line = label_width + 10
        svg_elements.append(
            f'<line x1="{start_x_line}" y1="{y_base}" x2="{start_x_line + track_area_width}" y2="{y_base}" stroke="#e2e8f0" stroke-width="1.5"/>'
        )

        merged_regions = merge_regions(track["regions"], max_gap=4)

        for reg in merged_regions:
            raw_start = reg.get("start", 1)
            raw_end = reg.get("end", 1)
            coord_str = f"{raw_start}-{raw_end}"

            p_start = max(1, min(raw_start, seq_length))
            p_end = max(1, min(raw_end, seq_length))

            x_pos = start_x_line + (p_start / seq_length) * track_area_width
            rect_w = max(((p_end - p_start) / seq_length) * track_area_width, 3)

            svg_elements.append(
                f'<rect class="lcr-region-rect" x="{x_pos:.1f}" y="{y_base - 4}" width="{rect_w:.1f}" height="10" '
                f'fill="{track["color"]}" rx="1">'
                f'<title>{track["label"]}: {coord_str}</title></rect>'
            )

    ruler_y = top_offset + (len(tracks) * row_height) + 6
    start_x_line = label_width + 10
    svg_elements.append(
        f'<line x1="{start_x_line}" y1="{ruler_y}" x2="{start_x_line + track_area_width}" y2="{ruler_y}" stroke="#334155" stroke-width="1.5"/>'
    )

    tick_step = 50 if seq_length <= 350 else (100 if seq_length <= 1000 else 200)
    curr_tick = 0
    while curr_tick <= seq_length:
        x_tick = start_x_line + (curr_tick / seq_length) * track_area_width
        svg_elements.append(
            f'<line x1="{x_tick:.1f}" y1="{ruler_y}" x2="{x_tick:.1f}" y2="{ruler_y + 5}" stroke="#334155" stroke-width="1.5"/>'
        )
        svg_elements.append(
            f'<text x="{x_tick:.1f}" y="{ruler_y + 18}" fill="#64748b" font-size="10" text-anchor="middle" font-family="sans-serif">{curr_tick}</text>'
        )
        curr_tick += tick_step

    return (
        f'<svg id="svg-{uniprot_id}" width="100%" height="{total_height}" viewBox="0 0 {total_width} {total_height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display: block; max-width: 100%; height: auto;">'
        f'{"".join(svg_elements)}</svg>'
    )


def render_record_rows(item: Dict[str, Any]) -> str:
    """Render individual table row for a distinct LCR entry with SVG track subrow."""
    protein_name = item.get("protein_name") or "Unknown"
    organism = item.get("organism") or "Unspecified"

    uni_data = fetch_uniprot_metadata(protein_name, organism)
    uniprot_id = uni_data["uniprot_id"]
    gene_name = uni_data["gene_name"]
    full_name = uni_data["full_name"]
    length = uni_data["length"]
    sequence = uni_data["sequence"]
    go_terms = uni_data.get("go_terms", [])

    platoloco_methods = {}
    if sequence:
        platoloco_methods = query_platoloco(sequence, header=uniprot_id)

    lcr_type_val = (
        item.get("binding_target")
        or item.get("proposed_function")
        or "Unspecified"
    )

    annot_start = parse_coord(item.get("start_of_annotation"))
    annot_end = parse_coord(item.get("end_of_annotation"))

    start_annot_str = str(annot_start) if annot_start is not None else "Unspecified"
    end_annot_str = str(annot_end) if annot_end is not None else "Unspecified"

    svg_track = generate_platoloco_style_svg(
        length,
        annot_start,
        annot_end,
        platoloco_methods,
        uniprot_id=uniprot_id,
    )

    uniprot_link = (
        f"https://www.uniprot.org/uniprotkb/{uniprot_id}"
        if uniprot_id != "N/A"
        else "#"
    )

    evidence_text = item.get("evidence") or "No evidence statement provided."
    source_id = item.get("doi") or item.get("file") or item.get("paper_id") or "N/A"

    category = (
        item.get("proposed_function")
        or item.get("binding_target")
        or "Unspecified"
    )
    go_ontology_str = "<br>".join(go_terms) if go_terms else "N/A"

    return f"""
            <tr>
                <td><a href="{uniprot_link}" target="_blank" class="protein-id">{uniprot_id}</a></td>
                <td><strong>{gene_name}</strong></td>
                <td>{full_name}</td>
                <td>{length if length > 0 else 'N/A'}</td>
                <td><span class="badge-type">{lcr_type_val}</span></td>
                <td><i>{organism}</i></td>
                <td><blockquote class="evidence-quote">"{evidence_text}"</blockquote></td>
                <td><code>{source_id}</code></td>
                <td><span class="badge-annot">{start_annot_str}</span></td>
                <td><span class="badge-annot">{end_annot_str}</span></td>
                <td><strong>{category}</strong></td>
                <td>{go_ontology_str}</td>
            </tr>
            <tr class="subrow">
                <td colspan="12" class="viz-container">
                    {svg_track}
                </td>
            </tr>
"""


def render_report_section(
    banner_text: str,
    records: List[Dict[str, Any]],
    headers: List[str],
    warning: bool = False,
) -> str:
    """Render one report section (banner + table) for a list of records."""
    if not records:
        return ""

    banner_class = "section-banner warning" if warning else "section-banner"
    header_cells = "".join(f"<th>{label}</th>" for label in headers)
    body_rows = "".join(render_record_rows(item) for item in records)

    return f"""
        <div class="{banner_class}">
            <span>{banner_text}</span>
            <span>{len(records)} entries</span>
        </div>
        <div class="table-wrapper">
        <table>
            <thead>
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
{body_rows}
            </tbody>
        </table>
        </div>
"""


def generate_html_report(
    paper_id: Optional[str] = None,
    output_html: Optional[str] = None,
) -> Optional[str]:
    """Generate structured HTML report dividing SQLite records into Verified and Manual Check sections."""
    records = fetch_records_from_db(paper_id=paper_id)
    if not records:
        print("Warning: No records found in SQLite database.")
        return None

    if not output_html:
        reports_dir = PROJECT_ROOT / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{paper_id}_report.html" if paper_id else "lcr_biocuration_report.html"
        output_html = str(reports_dir / filename)

    verified_records = []
    manual_check_records = []

    for item in records:
        status = str(item.get("curation_status", "")).lower()
        st_val = parse_coord(item.get("start_of_annotation"))
        end_val = parse_coord(item.get("end_of_annotation"))

        if status == "verified" and st_val is not None and end_val is not None:
            verified_records.append(item)
        else:
            manual_check_records.append(item)

    headers = [
        "UniprotID",
        "Gene name",
        "Name",
        "Protein length",
        "LCR type",
        "Organism",
        "Source",
        "Source ID",
        "Start of annotation",
        "End of annotation",
        "Annotation Category",
        "Gene Ontology of category",
    ]

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Functional LCR Annotation Report</title>
    <style>
        * { box-sizing: border-box; }
        html, body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            background-color: #f8fafc; 
            color: #334155; 
            margin: 0; 
            padding: 24px; 
            font-size: 13px; 
            width: 100%;
        }
        .container { 
            width: 100%; 
            max-width: 100%; 
            margin: 0 auto; 
        }
        header { margin-bottom: 24px; }
        h1 { font-size: 24px; color: #0f172a; margin: 0 0 4px 0; font-weight: 700; }
        .subtitle { color: #64748b; margin: 0; font-size: 13px; }
        
        .section-banner { 
            background: #1e293b; 
            color: #ffffff; 
            padding: 12px 16px; 
            border-radius: 8px 8px 0 0; 
            margin-top: 32px; 
            font-size: 13px; 
            font-weight: 600; 
            display: flex; 
            align-items: center; 
            justify-content: space-between; 
        }
        .section-banner.warning { background: #d97706; }
        
        .table-wrapper {
            width: 100%;
            overflow-x: auto;
            background: #ffffff;
            border-radius: 0 0 8px 8px;
            border: 1px solid #e2e8f0;
            border-top: none;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
            margin-bottom: 24px;
        }

        table { 
            width: 100%; 
            border-collapse: collapse; 
            table-layout: fixed; 
        }
        th, td { 
            padding: 12px 10px; 
            text-align: left; 
            border-bottom: 1px solid #f1f5f9; 
            vertical-align: top; 
            word-wrap: break-word; 
            overflow-wrap: break-word; 
        }
        th { 
            background-color: #f8fafc; 
            color: #475569; 
            font-weight: 600; 
            font-size: 11px; 
            text-transform: uppercase; 
            letter-spacing: 0.5px; 
            border-bottom: 2px solid #e2e8f0; 
        }
        tr:hover td { background-color: #f8fafc; }
        
        /* Fixed column width distribution to prevent awkward wrapping */
        th:nth-child(1), td:nth-child(1) { width: 7%; }
        th:nth-child(2), td:nth-child(2) { width: 7%; }
        th:nth-child(3), td:nth-child(3) { width: 10%; }
        th:nth-child(4), td:nth-child(4) { width: 5%; }
        th:nth-child(5), td:nth-child(5) { width: 8%; }
        th:nth-child(6), td:nth-child(6) { width: 8%; }
        th:nth-child(7), td:nth-child(7) { width: 18%; }
        th:nth-child(8), td:nth-child(8) { width: 10%; }
        th:nth-child(9), td:nth-child(9) { width: 6%; }
        th:nth-child(10), td:nth-child(10) { width: 6%; }
        th:nth-child(11), td:nth-child(11) { width: 8%; }
        th:nth-child(12), td:nth-child(12) { width: 7%; }

        .protein-id { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-weight: 600; color: #2563eb; text-decoration: none; }
        .protein-id:hover { text-decoration: underline; }
        .evidence-quote { font-style: italic; color: #475569; margin: 0; border-left: 3px solid #cbd5e1; padding-left: 8px; line-height: 1.4; font-size: 12px; }
        .badge-annot { background: #ffedd5; color: #c2410c; padding: 2px 4px; border-radius: 4px; font-weight: 600; font-family: monospace; font-size: 11px; display: inline-block; }
        .badge-type { background: #e0f2fe; color: #0369a1; padding: 3px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; display: inline-block; }
        .subrow { background-color: #f8fafc; border-bottom: 2px solid #cbd5e1; }
        .viz-container { padding: 16px; text-align: left; }
        
        .lcr-region-rect { cursor: pointer; transition: opacity 0.15s ease-in-out; }
        .lcr-region-rect:hover { opacity: 0.75; stroke: #0f172a; stroke-width: 1px; }

        @media print {
            @page { size: A4 landscape; margin: 10mm; }
            body { background-color: #ffffff !important; padding: 0 !important; width: 100% !important; font-size: 10px !important; }
            .table-wrapper { overflow: visible !important; border: none !important; box-shadow: none !important; }
            table { width: 100% !important; table-layout: fixed !important; }
            th, td { padding: 6px 4px !important; font-size: 9px !important; }
            svg { width: 100% !important; height: auto !important; max-width: 100% !important; page-break-inside: avoid; }
            tr.subrow { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Functional LCR Annotation Report</h1>
            <p class="subtitle">Structured low-complexity region dataset integrated with UniProt and PlaToLoCo predictions</p>
        </header>
"""

    html_content += render_report_section(
        "1. Verified LCRs (Experimental Binding & Coordinates)",
        verified_records,
        headers=headers,
    )
    html_content += render_report_section(
        "2. Requires Manual Check (Qualitative Mentions or Missing Coordinates)",
        manual_check_records,
        headers=headers,
        warning=True,
    )

    html_content += """
    </div>
</body>
</html>
"""

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Success! Report saved to: {output_html}")
    return output_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the LCR biocuration HTML report.")
    parser.add_argument("--paper-id", help="Filter report by specific paper ID or DOI.")
    parser.add_argument(
        "--output",
        help="Path where the HTML report will be written.",
    )
    args = parser.parse_args()

    generate_html_report(paper_id=args.paper_id, output_html=args.output)
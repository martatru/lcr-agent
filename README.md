# LCR-Agent

An automated pipeline for extracting and biocurating experimentally verified Low Complexity Regions (LCRs/LCDs/IDRs/PLDs) and their functional binding interactions from scientific literature. It combines zero-token literature screening, structured multi-model LLM extraction via the Groq API, UniProt metadata enrichment, local multi-track PlaToLoCo sequence predictions, and synthetic benchmarking.

## Key Features

* **Zero-Token Literature Screening**: Queries PubMed through SearXNG using RegEx and pattern matching to filter candidates before LLM inference[cite: 15, 16].
* **Automated PDF Retrieval**: Fetches Open Access papers automatically via Unpaywall and Europe PMC APIs based on DOI/PMID[cite: 15].
* **Header-to-Header Section Isolation**: Extracts `Results`, `Discussion`, and `Conclusion` sections while stripping `Methods` and `References` to maximize context quality and eliminate token waste[cite: 2, 15].
* **Structured Biocuration & Multi-Model Cascade**: Enforces strict Pydantic/Instructor JSON Schema validation for protein names, organisms, binding targets, explicit coordinates, and verbatim source evidence using `openai/gpt-oss-120b` with fallbacks (`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`)[cite: 9, 15].
* **UniProt Metadata Enrichment**: Queries the UniProt REST API to fetch canonical protein accession numbers, gene names, protein lengths, descriptions, and Gene Ontology (GO) terms[cite: 15].
* **Local 8-Track PlaToLoCo Integration**: Interfaces with a self-hosted PlaToLoCo Docker container API running all 8 predictor methods (`SEG`, `SEG-intermediate`, `SEG-strict`, `CAST`, `fLPS`, `fLPS-strict`, `SIMPLE`, `GBSC`)[cite: 15].
* **Synthetic Leakage-Free Benchmarking**: Evaluates LLM spatial math deduction, distractor domain handling, and negative control filtering using a 12-paper synthetic dataset runner (`run_synthetic_dataset.py`)[cite: 9, 15].
* **Interactive Dashboard & ZIP Export**: Generates an interactive 12-column HTML dashboard featuring native-style PlaToLoCo multi-track SVG visualizers with hover tooltips, segregating `Verified` records from `Requires Manual Check` entries with 1-click ZIP export[cite: 15].

## Project Structure

```text
lcr-agent/
├── data/
│   ├── debug/                 # Diagnostic logs and chunk data
│   ├── processed/             # Output files (JSONL, JSON, CSV, HTML reports)
│   └── raw_pdfs/              # Input directory for target PDF papers
├── docker/
│   └── platoloco/             # Local Git submodule for PlaToLoCo metaserver
├── searxng/                   # Local SearXNG configuration
├── src/
│   ├── diagnose_api.py        # Diagnostic runner for PlaToLoCo REST API endpoints
│   ├── generate_report.py     # HTML report generator with SVG visualizers & ZIP exporter
│   ├── llm_client.py          # LightLLMClient with multi-model fallback cascade
│   ├── main.py                # Core pipeline orchestration and prompt definition
│   ├── pdf_parser.py          # PDF text extraction utilities
│   ├── platoloco_client.py    # REST client for self-hosted PlaToLoCo API
│   ├── post_processor.py     # Programmatic consolidation and evidence merging
│   ├── run_synthetic_dataset.py # Evaluation script for synthetic dataset benchmarking
│   ├── text_processor.py      # Section segmentation, hyphenation repair, and verbatim verification
│   └── validator.py           # Verification rules for extracted LCR annotations
├── .env                       # Environment variables (API keys)
├── .gitignore                 # Version control rules
├── .gitmodules                # Git submodule mappings
├── docker-compose.yml         # Container configuration for PlaToLoCo and SearXNG
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
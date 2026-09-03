 # LCR-Agent

An automated pipeline for extracting and biocurating experimentally verified Low Complexity Regions (LCRs/LCDs/IDRs/PLDs) and their binding interactions from scientific literature. It combines literature screening with structured extraction using the Groq API, `instructor`, UniProt metadata integration, and local PlaToLoCo API sequence visualizers.

## Key Features

* **Automated PDF Retrieval**: Fetches Open Access papers automatically via Unpaywall and Europe PMC APIs based on DOI/PMID.
* **Structured Biocuration**: Enforces strict JSON Schema validation via Pydantic and Instructor for protein names, sequence ranges, and verbatim evidence.
* **UniProt Metadata Integration**: Automatically queries the UniProt REST API to fetch canonical protein identifiers, gene names, full protein descriptions, sequence lengths, and Gene Ontology (GO) terms.
* **Local PlaToLoCo Integration**: Interfaces with a self-hosted PlaToLoCo Docker container API to run multi-track LCR predictions (`SEG-intermediate`, `SEG-strict`, `CAST`, `fLPS`, and `fLPS-strict`) for sequence visualization.
* **Interactive HTML Dashboard**: Generates an advanced HTML report featuring multi-track SVG visualizers matching PlaToLoCo's native UI style, grouping records with specified ranges first and isolating unspecified ranges at the bottom.

## Project Structure

```text
lcr-agent/
├── data/
│   ├── debug/                 # Diagnostic logs and chunk data
│   ├── processed/             # Final outputs (JSONL, JSON, CSV, HTML reports)
│   └── raw_pdfs/              # Input directory for target PDF papers
├── src/
│   ├── diagnose.py            # Diagnostic script for PlaToLoCo predictor methods
│   ├── generate_report.py     # HTML report generator with UniProt & PlaToLoCo SVG tracks
│   ├── main.py                # Core pipeline orchestration for PDF processing
│   ├── platoloco_client.py    # Client for self-hosted PlaToLoCo API with graceful fallback
│   └── ...                    # Supporting parser, validator, and processor modules
├── docker-compose.yml         # Container configuration for local PlaToLoCo API
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```
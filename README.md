# LCR-Agent

An automated pipeline for extracting and biocurating experimentally verified Low Complexity Regions (LCRs/LCDs/IDRs/PLDs) and their binding interactions from scientific literature. It combines zero-token candidate screening via SearXNG with structured extraction using the Groq API (multi-model cascade) and `instructor`.

## Key Features

- **Zero-Token Literature Screening**: Queries PubMed through SearXNG using RegEx and pattern matching to filter candidates before running LLM inference.
- **Automated PDF Retrieval**: Fetches Open Access papers automatically via Unpaywall and Europe PMC APIs based on DOI/PMID.
- **Structured Biocuration**: Enforces strict JSON Schema validation via Pydantic and Instructor for protein names, sequence ranges, and verbatim evidence.
- **Robust Text Processing**: Isolates Results/Discussion sections, chunks text to manage rate limits, and strictly verifies LLM output against source text.
- **HTML Dashboard**: Generates an interactive HTML report segregating verified LCRs from qualitative mentions, featuring direct UniProt and DOI links.

## Project Structure

```text
lcr-agent/
├── data/
│   ├── debug/                 # Diagnostic logs and chunk data
│   ├── processed/             # Final outputs (JSONL, JSON, CSV, HTML)
│   └── raw_pdfs/              # Input directory for target PDF papers
├── download_pdfs.py           # Automated PDF retrieval via Unpaywall & Europe PMC
├── generate_report.py         # HTML report generator with UniProt & DOI links
├── llm_client.py              # Groq API client with multi-model fallback & Pydantic schema
├── main.py                    # Core pipeline orchestration for PDF processing
├── pdf_parser.py              # Zero-OCR text extraction using PyMuPDF
├── post_processor.py          # Post-processor creating clean JSON/CSV splits
├── searxng_finder.py          # 0-token literature search & RegEx pre-filtering
├── text_processor.py          # Section isolation, text cleaning, and chunking
├── validator.py               # Verbatim evidence validation against source text
└── README.md                  # Project documentation
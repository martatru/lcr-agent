# LCR-Agent

An automated pipeline for extracting and biocurating experimentally verified Low Complexity Regions (LCRs/LCDs/IDRs/PLDs) and their binding interactions from scientific literature. It combines zero-token candidate screening via SearXNG with structured extraction using the Groq API (`qwen/qwen3.8-27b`) and `instructor`.

## Key Features

- **Zero-Token Literature Screening**: Queries PubMed through SearXNG using RegEx and pattern matching (LCR terms, binding verbs, residue coordinates) to filter candidates before running LLM inference.
- **Structured Biocuration**: Enforces strict JSON Schema validation via Pydantic and Instructor for protein names, organisms, sequence ranges, verbatim evidence, and curator notes.
- **Robust Text Processing**: Handles hyphenation repair, reference section removal, rate-limit chunking, and verbatim sentence verification against source text.
- **HTML Dashboard**: Generates an interactive HTML report segregating verified LCRs from qualitative mentions, featuring direct UniProt and DOI links.

## Project Structure

```text
lcr-agent/
├── data/
│   ├── debug/                 # Diagnostic logs and chunk data
│   ├── processed/             # Final outputs (JSONL, JSON, CSV, HTML)
│   └── raw_pdfs/              # Input directory for target PDF papers
├── generate_report.py         # HTML report generator with UniProt & DOI links
├── llm_client.py              # Groq API client with Instructor & Pydantic schema
├── main.py                    # Core pipeline for PDF processing
├── pdf_parser.py              # Text extraction using PyMuPDF
├── process_results.py         # Post-processor creating clean JSON/CSV splits
├── searxng_finder.py          # 0-token literature search & RegEx pre-filtering
├── text_processor.py          # Text cleaning, normalization, and chunking
├── validator.py               # Verbatim evidence validation against source text
├── .env                       # API keys
├── README.md
└── requirements.txt
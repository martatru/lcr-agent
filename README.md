# LCR-Agent

An automated pipeline for extracting experimentally verified Low Complexity Regions (LCRs) from scientific PDF papers using the Google Gemini API and Pydantic structured validation.

## Project Structure

```text
lcr-agent/
├── data/
│   ├── annotated_lcr/     # Reference dataset files (.csv)
│   ├── debug/             # Diagnostic logs and extraction data per PDF
│   ├── processed/         # Final structured outputs (final_results.jsonl)
│   └── raw_pdfs/          # Input directory for target PDF papers
├── src/
│   ├── __pycache__/
│   ├── llm_client.py      # Gemini API communication and schema handling
│   ├── main.py            # Main asynchronous execution workflow
│   ├── pdf_parser.py      # PDF text extraction utilities
│   ├── text_processor.py  # Text cleaning, normalization, and chunking
│   └── validator.py       # Verification of extracted evidence against source text
├── venv/                  # Python virtual environment
├── README.md
└── requirements.txt
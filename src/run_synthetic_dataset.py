"""
Synthetic Dataset Multi-Model Runner for LCR-Agent Pipeline.

Processes synthetic_dataset.json across selected LLM models,
saves model-specific output JSON files, and triggers HTML report generation
for manual visual evaluation.
"""

import asyncio
import json
import logging
from pathlib import Path

from generate_report import generate_html_report
from llm_client import LightLLMClient
from main import PROMPT_LCR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of target models to evaluate
TARGET_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


async def run_model_evaluation(
    model_name: str, 
    dataset_file: str = "data/processed/synthetic_dataset.json"
) -> None:
    """Executes dataset extraction for a single model and renders an isolated HTML report."""
    safe_tag = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    output_json = Path(f"data/processed/verified_lcrs_{safe_tag}.json")
    output_html = Path(f"data/processed/lcr_report_{safe_tag}.html")

    input_path = Path(dataset_file)
    if not input_path.exists():
        logger.error("Dataset file '%s' not found!", dataset_file)
        return

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    papers = dataset.get("papers", [])
    if not papers:
        logger.warning("No papers found in %s", dataset_file)
        return

    # Configure client to prioritize the specified model
    client = LightLLMClient(max_concurrent=1)
    client.models = [model_name]

    results = []
    logger.info("==================================================")
    logger.info("STARTING EVALUATION FOR MODEL: %s", model_name)
    logger.info("==================================================")

    for paper in papers:
        paper_id = paper.get("paper_id", "Unknown_ID")
        difficulty = paper.get("difficulty_level", "Unknown Level")

        sections = paper.get("sections", {})
        results_text = sections.get("results", "")
        discussion_text = sections.get("discussion", "")
        full_text = paper.get("full_text") or f"Results\n{results_text}\n\nDiscussion\n{discussion_text}"

        logger.info("[%s] Processing: %s (%s)...", model_name, paper_id, difficulty)

        try:
            annotations = await client.generate_lcr_annotations(PROMPT_LCR, full_text)
            if annotations:
                results.append({
                    "file": f"{paper_id} ({difficulty})",
                    "annotations": annotations,
                })
        except Exception as err:
            logger.error("Failed processing %s with %s: %s", paper_id, model_name, err)

    # Save output JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Saved extracted annotations to %s", output_json)

    # Generate model-specific HTML report
    logger.info("Generating HTML report: %s", output_html)
    try:
        generate_html_report(input_file=str(output_json), output_html=str(output_html))
        logger.info("Finished report generation for %s\n", model_name)
    except Exception as report_err:
        logger.error("Failed generating report for %s: %s", model_name, report_err)


async def main() -> None:
    """Sequentially evaluates all target models."""
    dataset_path = "data/processed/synthetic_dataset.json"
    for model in TARGET_MODELS:
        await run_model_evaluation(model, dataset_file=dataset_path)


if __name__ == "__main__":
    asyncio.run(main())
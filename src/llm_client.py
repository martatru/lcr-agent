"""
LLM Client module using Instructor and Groq for structured LCR extraction.
"""

import os
import logging
import asyncio
import instructor
from groq import AsyncGroq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LCRAttribute(BaseModel):
    """Schema for individual Low Complexity Region (LCR) annotations."""
    protein_name: str = Field(
        description="Name or symbol of the protein, e.g., FUS, hnRNPA1, TIA1, Sup35"
    )
    organism: str = Field(
        default="Unspecified",
        description="Organism or species if mentioned, e.g., Human, Mouse, Yeast"
    )
    start_of_annotation: str = Field(
        default="Unspecified",
        description="Explicit numerical start residue position if stated in text (e.g., '904'), otherwise 'Unspecified'"
    )
    end_of_annotation: str = Field(
        default="Unspecified",
        description="Explicit numerical end residue position if stated in text (e.g., '1297'), otherwise 'Unspecified'"
    )
    binding_target: str = Field(
        default="Unspecified",
        description="Precise interaction type using the suffix '-binding', e.g., 'protein-binding', 'RNA-binding', 'DNA-binding', 'lipid-binding', or 'Unspecified'. Never use isolated nouns."
    )
    proposed_function: str = Field(
        description="Specific molecular binding function or phase transition described in text"
    )
    evidence: str = Field(
        description="Exact verbatim sentence from the text as proof of LCR binding or interaction"
    )
    curation_status: str = Field(
        default="Requires Manual Check",
        description="Must be 'Verified' if experimental binding and coordinates are present; otherwise 'Requires Manual Check'"
    )
    curator_note: str = Field(
        default="Unspecified",
        description="Flag or suggestion for biocuration, e.g. 'Exact positions given' or 'Qualitative mention only'"
    )


class LCRResponse(BaseModel):
    """Container for a list of extracted LCR annotations."""
    annotations: list[LCRAttribute] = Field(
        default_factory=list,
        description="List of extracted LCR annotations"
    )


class LightLLMClient:
    """Asynchronous Groq API client with multi-model fallback cascade."""

    def __init__(self, max_concurrent: int = 1):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable missing!")

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = instructor.from_groq(
            AsyncGroq(api_key=api_key),
            mode=instructor.Mode.MD_JSON
        )
        self.models = [
            "openai/gpt-oss-120b",
            # "openai/gpt-oss-20b"
        ]

    async def generate_lcr_annotations(self, prompt: str, text: str) -> list[dict]:
        """Generates structured LCR annotations, failing over to backup models on rate limits."""
        async with self.semaphore:
            for model_name in self.models:
                for attempt in range(1, 4):
                    try:
                        response: LCRResponse = await self.client.chat.completions.create(
                            model=model_name,
                            response_model=LCRResponse,
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": f"DOCUMENT TEXT:\n{text}"}
                            ],
                            temperature=0.0,
                            max_tokens=4096,
                            max_retries=2
                        )
                        return [ann.model_dump() for ann in response.annotations]

                    except Exception as error:
                        err_str = str(error)
                        if "429" in err_str and "TPD" in err_str:
                            logger.warning(
                                "Daily limit (TPD) reached for %s. Cascading to next backup model...",
                                model_name
                            )
                            break
                        elif "429" in err_str:
                            logger.warning(
                                "Minute limit (TPM) hit on %s. Pausing 10s (Attempt %d/3)...",
                                model_name, attempt
                            )
                            await asyncio.sleep(10)
                        else:
                            logger.error("Groq API Error on model %s: %s", model_name, error)
                            break
            return []
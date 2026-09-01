import os
import logging
import asyncio
import instructor
from groq import AsyncGroq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LCRAttribute(BaseModel):
    protein_name: str = Field(description="Name or symbol of the protein, e.g., FUS, hnRNPA1, TIA1, Sup35")
    organism: str = Field(default="Unspecified", description="Organism or species if mentioned, e.g., Human, Mouse, Yeast")
    start_of_annotation: str = Field(default="Unspecified", description="Numerical start position if explicitly stated (e.g. '2'), otherwise 'Unspecified'")
    end_of_annotation: str = Field(default="Unspecified", description="Numerical end position if explicitly stated (e.g. '214'), otherwise 'Unspecified'")
    proposed_function: str = Field(description="Biological/molecular function or transition state described in text")
    evidence: str = Field(description="Exact verbatim sentence from the text as proof")
    curator_note: str = Field(description="Flag/Suggestion for biocuration, e.g. 'Exact positions given' or 'Qualitative mention only - check UniProt for canonical boundaries'")

class LCRResponse(BaseModel):
    annotations: list[LCRAttribute] = Field(default_factory=list, description="List of extracted LCRs")

class LightLLMClient:
    def __init__(self, max_concurrent: int = 1):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable missing!")
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = instructor.from_groq(AsyncGroq(api_key=api_key), mode=instructor.Mode.JSON)

    async def generate_lcr_annotations(self, prompt: str, text: str) -> list[dict]:
        async with self.semaphore:
            try:
                response: LCRResponse = await self.client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    response_model=LCRResponse,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"DOCUMENT TEXT:\n{text}"}
                    ],
                    temperature=0.0,
                    max_retries=3
                )
                return [ann.model_dump() for ann in response.annotations]
            except Exception as error:
                logger.error(f"Groq API Error: {error}")
                return []
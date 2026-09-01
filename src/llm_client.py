import os
import logging
import asyncio
import instructor
from groq import AsyncGroq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LCRAttribute(BaseModel):
    start_of_annotation: int = Field(description="Explicit numerical start residue position, e.g. 120")
    end_of_annotation: int = Field(description="Explicit numerical end residue position, e.g. 150")
    proposed_function: str = Field(description="Explicitly stated biological or molecular function")
    evidence: str = Field(description="Exact verbatim sentence from the text as proof")

class LCRResponse(BaseModel):
    annotations: list[LCRAttribute] = Field(default_factory=list, description="List of extracted LCRs")

class LightLLMClient:
    def __init__(self, max_concurrent: int = 1):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable missing!")
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        # Instructor wymusza poprawny schemat Pydantic z LLM
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
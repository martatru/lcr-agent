import os
import json
import logging
import asyncio
import random
import re
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class LCRAttribute(BaseModel):
    start_of_annotation: int = Field(description="Explicit numerical start residue position")
    end_of_annotation: int = Field(description="Explicit numerical end residue position")
    proposed_function: str = Field(description="Explicitly stated biological or molecular function")
    evidence: str = Field(description="Exact verbatim sentence from the text as proof")


class LightLLMClient:
    def __init__(self, max_concurrent: int = 1):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable missing. Remember to run export GEMINI_API_KEY=...")
        
        self.client = genai.Client(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    async def generate_lcr_annotations(self, prompt: str, text: str) -> list[dict]:
        async with self.semaphore:
            max_retries = 5
            formatted_contents = f"DOCUMENT TEXT:\n{text}\n\nINSTRUCTIONS AND TASK:\n{prompt}"

            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=list[LCRAttribute],
                safety_settings=self.safety_settings,
            )

            for attempt in range(max_retries):
                try:
                    response = await self.client.aio.models.generate_content(
                        model='gemini-3.5-flash-lite',
                        contents=formatted_contents,
                        config=config,
                    )
                    
                    raw_output = response.text
                    if not raw_output:
                        logger.warning("Empty response from API. Retrying (%d/%d)", attempt + 1, max_retries)
                        await asyncio.sleep(5)
                        continue

                    return self._parse_json_safely(raw_output)

                except Exception as error:
                    error_str = str(error)
                    wait_time = 12 + random.uniform(2.0, 5.0)
                    logger.warning(
                        "API error (%s...). Waiting %.1fs... (Attempt %d/%d)",
                        error_str[:50], wait_time, attempt + 1, max_retries
                    )
                    await asyncio.sleep(wait_time)
            
            logger.error("Too many blocks or errors. Skipping this chunk.")
            return []

    def _parse_json_safely(self, raw_string: str) -> list[dict]:
        if not raw_string:
            return []

        match = re.search(r"(\[.*\]|\{.*\})", raw_string, re.DOTALL)
        if not match:
            logger.error("No valid JSON block found in the response.")
            return []

        clean_string = match.group(1)

        try:
            data = json.loads(clean_string)
            return [data] if isinstance(data, dict) else data
        except json.JSONDecodeError as error:
            logger.error("JSON decoding error: %s", error)
            return []
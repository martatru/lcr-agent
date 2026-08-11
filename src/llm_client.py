import json
import logging
import aiohttp

logger = logging.getLogger(__name__)


class LightLLMClient:
    """Async client for querying the LightLLM API."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }

    async def generate_lcr_annotations(self, prompt: str, text: str) -> list[dict]:
        """
        Sends the processed text and system prompt to LightLLM.
        Expects a JSON array in response.
        """
        url = f"{self.base_url}/generate"
        payload = {
            "inputs": f"{prompt}\n\nDOCUMENT TEXT:\n{text}",
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.1,  # Keep low to minimize hallucinations
                "do_sample": False
            }
        }

        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    raw_output = result.get("generated_text", "[]")

                    # Attempt to parse the LLM string output into a Python list
                    return self._parse_json_safely(raw_output)

            except aiohttp.ClientError as error:
                logger.error("LLM API network error: %s", error)
                return []

    def _parse_json_safely(self, raw_string: str) -> list[dict]:
        """Safely parses JSON, stripping potential markdown formatting."""
        clean_string = raw_string.strip()
        
        # Remove markdown code blocks if the LLM hallucinated them
        if clean_string.startswith("```json"):
            clean_string = clean_string[7:]
        if clean_string.endswith("```"):
            clean_string = clean_string[:-3]
            
        try:
            return json.loads(clean_string.strip())
        except json.JSONDecodeError as error:
            logger.error("LLM did not return a valid JSON array: %s", error)
            return []
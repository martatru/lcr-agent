import os
import logging
import aiohttp

logger = logging.getLogger(__name__)


class DifyClient:
    """Async client for handling PDF extraction via Dify API."""

    def __init__(self, api_key: str, base_url: str = "https://api.dify.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    async def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Uploads a PDF to Dify and returns the extracted raw text.
        """
        upload_url = f"{self.base_url}/files/upload"

        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                with open(file_path, "rb") as file:
                    form_data = aiohttp.FormData()
                    form_data.add_field(
                        "file",
                        file,
                        filename=os.path.basename(file_path)
                    )
                    # Dify requires a user identifier for uploads
                    form_data.add_field("user", "lcr-agent-system")

                    async with session.post(upload_url, data=form_data) as response:
                        response.raise_for_status()
                        result = await response.json()

                # Adapt this depending on whether you use a Dify Workflow 
                # or just raw document extraction API.
                return result.get("text", "")

            except aiohttp.ClientError as error:
                logger.error("Network error processing %s: %s", file_path, error)
                return ""
            except IOError as error:
                logger.error("File error reading %s: %s", file_path, error)
                return ""
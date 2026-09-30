"""
LLM Client module using Instructor and Groq for structured LCR extraction.
"""

import os
import re
import logging
import asyncio
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
import instructor
from groq import AsyncGroq

logger = logging.getLogger(__name__)

BindingTargetType = Literal[
    "protein-binding", "RNA-binding", "DNA-binding", "lipid-binding", "Unspecified"
]


class LCRAttribute(BaseModel):
    """Schema for individual Low Complexity Region (LCR) annotations."""
    protein_name: str = Field(
        description="Name or symbol of the protein, e.g., FUS, hnRNPA1, TIA1, Sup35"
    )
    organism: str = Field(
        default="Unspecified",
        description="Organism or species if mentioned, e.g., Human, Mouse, Yeast"
    )
    start_of_annotation: Optional[int] = Field(
        default=None,
        description="Explicit numerical start residue position if stated in text (e.g., 904), otherwise None"
    )
    end_of_annotation: Optional[int] = Field(
        default=None,
        description="Explicit numerical end residue position if stated in text (e.g., 1297), otherwise None"
    )
    binding_target: BindingTargetType = Field(
        default="Unspecified",
        description="Precise interaction type using the suffix '-binding', e.g., 'protein-binding', 'RNA-binding', 'DNA-binding', 'lipid-binding', or 'Unspecified'."
    )
    proposed_function: str = Field(
        description="Specific molecular binding function or phase transition described in text"
    )
    evidence: str = Field(
        description="Exact verbatim sentence from the text as proof of LCR binding or interaction"
    )
    evidence_verified: bool = Field(
        default=False,
        description="Whether evidence sentence was found verbatim in text chunk"
    )
    curation_status: str = Field(
        default="Requires Manual Check",
        description="Must be 'Verified' if experimental binding and coordinates are present; otherwise 'Requires Manual Check'"
    )


class LCRResponse(BaseModel):
    """Container for a list of extracted LCR annotations."""
    annotations: List[LCRAttribute] = Field(
        default_factory=list,
        description="List of extracted LCR annotations"
    )


def evidence_in_chunk(evidence: str, chunk: str) -> bool:
    """Check if the extracted evidence phrase exists verbatim in the text chunk."""
    if not evidence:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return norm(evidence) in norm(chunk)


def compute_curation_status(
    is_verified_evidence: bool,
    start: Optional[int],
    end: Optional[int],
    binding_target: str
) -> str:
    """Compute curation status based on objective code-level criteria."""
    if is_verified_evidence and start is not None and end is not None and binding_target != "Unspecified":
        return "Verified"
    return "Requires Manual Check"


class LightLLMClient:
    """Asynchronous Groq API client with multi-model fallback cascade."""

    def __init__(self, max_concurrent: int = 1):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable missing!")

        self.raw_client = AsyncGroq(api_key=api_key)
        self.client = instructor.from_groq(self.raw_client, mode=instructor.Mode.MD_JSON)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Original model configuration
        self.models = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b"
        ]
        self._valid_models: List[str] = []

    async def validate_models(self) -> List[str]:
        """Validate configured models against Groq API without inserting random models."""
        try:
            available = await self.raw_client.models.list()
            available_ids = {m.id for m in available.data}
            
            # Filter ONLY user-defined models
            self._valid_models = [m for m in self.models if m in available_ids]
            
            if not self._valid_models:
                logger.warning(
                    "Models %s not found on Groq account. Keeping original list for connection attempt.",
                    self.models
                )
                self._valid_models = list(self.models)
                
            return self._valid_models
        except Exception as err:
            logger.error("Error checking Groq models: %s", err)
            self._valid_models = list(self.models)
            return self.models

    async def generate_lcr_annotations(self, prompt: str, text: str) -> Dict[str, Any]:
        """Generates structured LCR annotations, flagging unverified evidence instead of deleting it."""
        if not self._valid_models:
            await self.validate_models()

        async with self.semaphore:
            last_error = None

            for model_name in self._valid_models:
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

                        annot_dicts = []
                        for ann in response.annotations:
                            ann_dict = ann.model_dump()
                            
                            # 1. Check if evidence is verbatim in text
                            verified = evidence_in_chunk(ann.evidence, text)
                            ann_dict["evidence_verified"] = verified
                            
                            # 2. Compute curation status programmatically
                            ann_dict["curation_status"] = compute_curation_status(
                                verified,
                                ann.start_of_annotation,
                                ann.end_of_annotation,
                                ann.binding_target
                            )
                            
                            if not verified:
                                logger.info(
                                    "Unverified evidence (flagged for manual check): %s", 
                                    ann.evidence[:60]
                                )

                            # KEEP ALL ANNOTATIONS (no deletion!)
                            annot_dicts.append(ann_dict)

                        if not annot_dicts:
                            return {"status": "empty", "annotations": [], "error": None}

                        return {"status": "ok", "annotations": annot_dicts, "error": None}

                    except Exception as error:
                        last_error = str(error)
                        if "429" in last_error and "TPD" in last_error:
                            logger.warning("Daily limit (TPD) reached for %s. Cascading to backup model...", model_name)
                            break
                        elif "429" in last_error:
                            logger.warning("Minute limit (TPM) hit on %s (Attempt %d/3). Pausing 10s...", model_name, attempt)
                            await asyncio.sleep(10)
                        else:
                            logger.error("Groq API error on model %s: %s", model_name, error)
                            break

            return {"status": "failed", "annotations": [], "error": last_error}
import re
import unicodedata

BINDING_KEYWORDS = [
    "bind", "binding", "bound", "interact", "interaction",
    "complex", "recruits", "associates", "recognizes", "partner"
]


def normalize_text(text: str) -> str:
    """Normalizes unicode characters and whitespace in text for exact matching."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def validate_lcr_annotations(annotations: list[dict], full_text: str) -> list[dict]:
    """
    Validates extracted LCR annotations against the source text.
    Verifies verbatim evidence presence and checks for binding interaction relevance.
    """
    normalized_full = normalize_text(full_text)
    valid_annotations = []

    for ann in annotations:
        evidence = ann.get("evidence", "")
        normalized_evidence = normalize_text(evidence)

        # Check if evidence exists in full text
        evidence_valid = bool(
            normalized_evidence and (
                normalized_evidence in normalized_full or
                normalized_evidence[:40] in normalized_full
            )
        )

        # Verify binding context in target, evidence, or function description
        binding_target = ann.get("binding_target", "").lower()
        function_desc = ann.get("proposed_function", "").lower()

        has_binding_context = (
            binding_target not in ["unspecified", "none", ""] or
            any(kw in normalized_evidence for kw in BINDING_KEYWORDS) or
            any(kw in function_desc for kw in BINDING_KEYWORDS)
        )

        if evidence_valid and has_binding_context:
            valid_annotations.append(ann)

    return valid_annotations
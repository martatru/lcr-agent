import re
import unicodedata

def normalize_text(text: str) -> str:
    # Remove Unicode ligatures (e.g., \ufb01 -> fi)
    text = unicodedata.normalize("NFKC", text)
    # Merge line breaks and multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def validate_lcr_annotations(annotations: list, full_text: str) -> list:
    normalized_full = normalize_text(full_text)
    valid = []

    for ann in annotations:
        evidence = ann.get("evidence", "")
        normalized_evidence = normalize_text(evidence)

        # Check for the presence of the full sentence or at least its first fragment
        if normalized_evidence and (
            normalized_evidence in normalized_full 
            or normalized_evidence[:40] in normalized_full
        ):
            valid.append(ann)

    return valid
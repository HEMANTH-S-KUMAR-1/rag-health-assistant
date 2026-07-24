"""
verify.py
---------
Citation verification: cross-check that specific numbers, dates, and
percentages in the generated answer actually appear in the retrieved
chunk text. Catches subtle hallucinations the prompt-level constraint
alone might miss.

Usage (as a library — called from cli.py):
    from verify import verify_citations
    claims = verify_citations(answer_text, retrieved_chunks)
"""
import re


def extract_claims(text: str) -> list:
    """
    Extract verifiable claims from generated text:
    numbers, percentages, currency amounts, dates/years, and
    named quantities (e.g. "1.86 lakh", "Rs. 5 lakh").
    """
    patterns = [
        # Currency amounts like "Rs. 5 lakh", "Rs. 1,80,435 crore"
        r"Rs\.?\s*[\d,]+(?:\.\d+)?\s*(?:lakh|crore|thousand)?",
        # Quantities with lakh/crore: "1.86 lakh", "44.14 crore"
        r"\d+(?:\.\d+)?\s*(?:lakh|crore)\b",
        # Percentages: "60%", "17.3%"
        r"\d+(?:\.\d+)?(?:\s*(?:per\s*cent|%))",
        # Years: 2014, 2018, 2024
        r"\b(?:19|20)\d{2}\b",
        # Standalone large numbers: "36,218", "220"
        r"\b\d{1,3}(?:,\d{2,3})+\b",
    ]

    claims = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claim = match.group().strip()
            if claim not in seen and len(claim) > 2:
                seen.add(claim)
                claims.append(claim)
    return claims


def normalize(text: str) -> str:
    """Normalize whitespace and punctuation for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def verify_citations(answer: str, retrieved_chunks: list) -> dict:
    """
    Check each extracted claim against the source chunk text.
    Returns a dict with verified and unverified claims.
    """
    claims = extract_claims(answer)
    if not claims:
        return {"verified": [], "unverified": [], "total": 0}

    # Build combined source text from retrieved chunks
    source_text = normalize(
        " ".join(r["chunk"]["text"] for r in retrieved_chunks)
    )

    verified = []
    unverified = []

    for claim in claims:
        normalized_claim = normalize(claim)
        if normalized_claim in source_text:
            verified.append(claim)
        else:
            # Try a looser match (just the numeric part)
            numbers = re.findall(r"[\d,]+(?:\.\d+)?", claim)
            if any(n in source_text for n in numbers if len(n) > 2):
                verified.append(claim)
            else:
                unverified.append(claim)

    return {
        "verified": verified,
        "unverified": unverified,
        "total": len(claims),
    }
